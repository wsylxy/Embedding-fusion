from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import register_model

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
    def __init__(self, dim=768, n_heads=8, hidden=1024,
                 use_attn_pool=True, dropout=0.1, out_dim=None) -> None:
        super().__init__()
        # 双向 cross-attn
        self.t2m = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.m2t = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        # 残差 + LN
        self.ln_t = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        # # 池化
        # self.pool_t = AttnPool(dim) if use_attn_pool else nn.Identity()
        # self.pool_m = AttnPool(dim) if use_attn_pool else nn.Identity()

        # # 句向量融合头：Concat-MLP + Gated Sum（轻量但表达力强）
        # self.mlp = nn.Sequential(
        #     nn.Linear(dim * 2, hidden),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden, self.out_dim)
        # )

        self.proj_m = nn.Linear(dim, dim)

        # gating: token-level + 非线性变换，增强表达能力
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()   # gate ∈ [0,1]
        )

        # self.out_ln = nn.LayerNorm(self.out_dim)
    def forward(self, H_t, H_m, mask_t=None, mask_m=None, normalize=True):
        # cross-attn
        mask_m = (mask_m==0)
        T2M, _ = self.t2m(H_t, H_m, H_m, key_padding_mask=mask_m)   # text attends math -> [B,Lt,D]
        e_t = self.ln_t(H_t + self.drop(T2M))  # [B,Lt,D]
        h_math_proj = self.proj_m(H_m)
        gate = self.gate(h_math_proj)
        e_m = h_math_proj * gate
        # 句向量融合：Concat-MLP + Gated Sum
        fused_mlp = torch.cat([e_t, e_m], dim=1)                         # [B,Lt+Lm,D]
        return fused_mlp


@register_model(name="fuse")
def build_model(cfg) -> nn.Module:
    return CrossAttnfuse(
        dim=768, 
        n_heads=8, 
        hidden=1024,
        use_attn_pool=True, 
        dropout=0.1, 
        out_dim=None
    )
