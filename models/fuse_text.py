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

class CrossAttnfuse_text(nn.Module):
    def __init__(self, dim=768, n_heads=8, hidden=1024,
                 use_attn_pool=True, dropout=0.1, out_dim=128) -> None:
        super().__init__()
        # 双向 cross-attn
        self.t2m = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        # 残差 + LN
        self.ln_t = nn.LayerNorm(dim)
        self.ln_m = nn.LayerNorm(dim)
        self.delta_proj = nn.Linear(dim, dim)
        self.ln_delta = nn.LayerNorm(dim)
        # nn.init.zeros_(self.delta_proj.weight)
        # nn.init.zeros_(self.delta_proj.bias)

        # gating: token-level + 非线性变换，增强表达能力
        self.gate_tok = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, 1), nn.Sigmoid()
        )

        self.out = nn.Linear(dim, out_dim)
        self._init_weights()

    def _init_weights(self):
        # delta_proj：不要全 0；用 xavier 更稳
        nn.init.xavier_uniform_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)

        # gate 的前一层正常初始化
        nn.init.xavier_uniform_(self.gate_tok[0].weight)
        nn.init.zeros_(self.gate_tok[0].bias)

        # gate 的最后一层：权重=0，偏置负大数，让 g≈0
        nn.init.zeros_(self.gate_tok[2].weight)        # shape [1, dim]
        nn.init.constant_(self.gate_tok[2].bias, -5.0) # sigmoid(-5)≈0.0067

    # def __init__(self, dim=768, n_heads=8, hidden=1024,
    #              use_attn_pool=True, dropout=0.1, out_dim=128) -> None:
    #     super().__init__()
    #     self.ln_t = nn.LayerNorm(dim)

        
    #     self.out = nn.Linear(dim, out_dim)
    
    def forward(self, H_t, H_m, mask_t=None, mask_m=None, normalize=True):
        # cross-attn
        mask_mr = (mask_m==0)
        T = self.ln_t(H_t)                       # [B, Lt, D]
        M = self.ln_m(H_m)                       # [B, Lm, D]
        delta_t, _ = self.t2m(T, M, M, key_padding_mask=mask_mr)   # text attends math -> [B,Lt,D]
        delta_t = self.delta_proj(delta_t)                               # [B, Lt, D]
        delta_t = self.ln_delta(delta_t)
        ############ FOR FORMULA ENCODER IS TRAINED WITH MEAN POOLING ################
        # m_len = mask_m.sum(dim=1, keepdim=True).clamp_min(1)          # [B,1]
        # m_sum = (M * mask_m.unsqueeze(-1)).sum(dim=1, keepdim=True)   # [B,1,D]
        # m_pooled = m_sum / m_len.unsqueeze(-1)     # [B,1,D]              
        # # 逐 token 门控（由 math 决定），也可改成用 pooled math 做全局门
        # g = self.gate_tok(m_pooled)               # [B,1,D]->[B,1,1]
        ##############################################################################

        ############ FOR FORMULA ENCODER IS TRAINED WITH MAXSIM ######################
        g = self.gate_tok(delta_t)               # [B,L,D]->[B,L,1]
        ##############################################################################
        # 句向量融合：Concat-MLP + Gated Sum
        T_fused = T + g * delta_t   # [B, Lt, D]
        Z = self.out(T_fused)
        out_mask = mask_t
        return Z, out_mask
    
    # def forward(self, H_t, H_m, mask_t=None, mask_m=None, normalize=True):
    #     # cross-attn
    #     mask_mr = (mask_m==0)
    #     T = self.ln_t(H_t)                       # [B, Lt, D]
    #     Z = self.out(T)
    #     out_mask = mask_t
    #     return Z, out_mask


@register_model(name="fuse_text")
def build_model(cfg) -> nn.Module:
    print("building model CrossAttnfuse_text")
    return CrossAttnfuse_text(
        dim=768, 
        n_heads=8, 
        hidden=1024,
        use_attn_pool=True, 
        dropout=0.1, 
        out_dim=128
    )
