from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from criterions.registry import register_criterion

# class AttnPool(nn.Module):
#     """attention pooling, [B,L,D] -> [B,D], mask(1=valid,0=pad)"""
#     def __init__(self, dim):
#         super().__init__()
#         self.q = nn.Linear(dim, 1, bias=False)

#     def forward(self, H, mask=None):
#         # H:[B,L,D], mask:[B,L] with 1/0
#         score = self.q(H).squeeze(-1)                # [B,L]
#         if mask is not None:
#             score = score.masked_fill(mask == 0, -1e9)
#         attn = torch.softmax(score, dim=-1)          # [B,L]
#         return torch.einsum('bl,bld->bd', attn, H)   # [B,D]

class CrossAttnfuse(nn.Module):
    def __init__(self, dim:int, n_heads=8, hidden=1024,
                 use_attn_pool=True, dropout=0.1, out_dim=None) -> None:
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or dim
        self.use_attn_pool = use_attn_pool
        # 双向 cross-attn
        self.t2m = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.m2t = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        # 残差 + LN
        self.ln_t = nn.LayerNorm(dim)
        self.ln_m = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        # 池化
        self.pool_t = AttnPool(dim) if use_attn_pool else nn.Identity()
        self.pool_m = AttnPool(dim) if use_attn_pool else nn.Identity()

        # 句向量融合头：Concat-MLP + Gated Sum（轻量但表达力强）
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.out_dim)
        )
        self.out_ln = nn.LayerNorm(self.out_dim)
    def forward(self, H_t, H_m, mask_t=None, mask_m=None, normalize=True):
        kpm_t = (mask_t == 0) if mask_t is not None else None  # True=pad
        kpm_m = (mask_m == 0) if mask_m is not None else None
        # 双向 cross-attn
        T2M, _ = self.t2m(H_t, H_m, H_m, key_padding_mask=kpm_m)   # text attends math -> [B,Lt,D]
        M2T, _ = self.m2t(H_m, H_t, H_t, key_padding_mask=kpm_t)   # math  attends text -> [B,Lm,D]
        # 残差 + LN
        T_ = self.ln_t(H_t + self.drop(T2M))  # [B,Lt,D]
        M_ = self.ln_m(H_m + self.drop(M2T))  # [B,Lm,D]
        # 池化成句向量
        e_t = (self.pool_t(T_, mask_t) if self.use_attn_pool else T_.mean(dim=1))  # [B,D]
        e_m = (self.pool_m(M_, mask_m) if self.use_attn_pool else M_.mean(dim=1))  # [B,D]
        mask_t = mask_t.unsqueeze(-1)
        mask_m = mask_m.unsqueeze(-1)
        e_t = e_t*mask_t
        e_m = e_m*mask_m
        lengths_t = mask_t.sum(dim=1)   #length of the valid part of each sample
        lengths_m = mask_m.sum(dim=1)
        e_t = e_t.sum(dim=1)/lengths_t.clamp(min=1) #avoid the valid length of a sample is 0
        e_m = e_m.sum(dim=1)/lengths_m.clamp(min=1)
        # 句向量融合：Concat-MLP + Gated Sum
        cat = torch.cat([e_t, e_m], dim=-1)                         # [B,2D]
        fused_mlp = self.mlp(cat) 
        return fused_mlp


@register_criterion(name="fuse")
def build_model(cfg) -> nn.Module:
    return CrossAttnfuse(
        dim=768, 
        n_heads=8, 
        hidden=1024,
        use_attn_pool=True, 
        dropout=0.1, 
        out_dim=None
    )
