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
                 use_attn_pool=True, dropout=0.1, out_dim=128) -> None:
        super().__init__()
        # 双向 cross-attn
        self.t2m = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.m2t = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        # 残差 + LN
        self.ln_t1 = nn.LayerNorm(dim)
        self.ln_t2 = nn.LayerNorm(dim)
        self.ln_m1 = nn.LayerNorm(dim)
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

        self.outln = nn.Linear(dim, out_dim)
        # self.out_ln = nn.LayerNorm(self.out_dim)
    def compact_by_mask(self, e_t, e_m, mask_t, mask_m):
        # e_t: [B,Lt,D], e_m: [B,Lm,D], mask_*: [B,L*], 1/True=valid
        B, _, D = e_t.shape
        # print('mask_t', mask_t)
        # print('mask_m', mask_m)
        fused = torch.cat([e_t, e_m], dim=1)                      # [B,Lt+Lm,D]
        fused_mask = torch.cat([mask_t, mask_m], dim=1).bool()    # [B,Lt+Lm]
        # print('fused_mask', fused_mask.shape)
        # 构造“稳定排序”的 key：先按 pad(=1) / valid(=0)，再按原始位置
        key1 = (~fused_mask).to(torch.int64)                      # valid→0, pad→1
        order = torch.arange(fused.size(1), device=fused.device).unsqueeze(0).expand_as(key1)
        scores = key1 * (fused.size(1) + 1) + order               # [B,L], 两关键字合一

        idx_2d = torch.argsort(scores, dim=1)                        # 稳定地把 valid 放前
        idx_3d = idx_2d.unsqueeze(-1).expand(-1, -1, D)                 # [B,L, D]
        # print('idx', idx_2d.shape)
        fused_compact = fused.gather(1, idx_3d)                      # [B,Lt+Lm,D]
        mask_compact  = fused_mask.gather(1, idx_2d)     # [B,Lt+Lm]
        # print('mask_compact', mask_compact)
        # （可选）裁到每条样本的有效长度上限，减少 pad
        L = mask_compact.sum(1)                                   # [B]
        Lmax = int(L.max().item())
        mask_compact = mask_compact.long()
        fused_compact = fused_compact[:, :Lmax]
        mask_compact  = mask_compact[:, :Lmax]
        # print('after mask_compact', mask_compact)
        # print('after mask_compact.shape', mask_compact.shape)
        return fused_compact, mask_compact
    
    def forward(self, H_t, H_m, mask_t=None, mask_m=None, normalize=True):
        # cross-attn
        mask_mr = (mask_m==0)
        T2M, _ = self.t2m(self.ln_t1(H_t), self.ln_m1(H_m), self.ln_m1(H_m), key_padding_mask=mask_mr)   # text attends math -> [B,Lt,D]
        e_t = self.ln_t2(H_t + self.drop(T2M))  # [B,Lt,D]
        h_math_proj = self.proj_m(H_m)
        gate = self.gate(h_math_proj)
        e_m = h_math_proj * gate
        # 句向量融合：Concat-MLP + Gated Sum
        # fused_mlp = torch.cat([e_t, e_m], dim=1)                         # [B,Lt+Lm,D]
        fused_compact, mask_compact = self.compact_by_mask(e_t, e_m, mask_t, mask_m)
        fused_compact = self.outln(fused_compact)
        return fused_compact, mask_compact


@register_model(name="fuse")
def build_model(cfg) -> nn.Module:
    return CrossAttnfuse(
        dim=768, 
        n_heads=8, 
        hidden=1024,
        use_attn_pool=True, 
        dropout=0.1, 
        out_dim=128
    )
