from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import register_model

class CrossAttnfuse_text(nn.Module):
    def __init__(self, dim=768, n_heads=8, hidden=1024,
                 use_attn_pool=True, dropout=0.1, out_dim=128) -> None:
        super().__init__()
        #  cross-attn
        self.t2m = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        # residual connection + LN
        self.ln_t = nn.LayerNorm(dim)
        self.ln_m = nn.LayerNorm(dim)
        self.delta_proj = nn.Linear(dim, dim)
        self.ln_delta = nn.LayerNorm(dim)
        bottleneck = 256
        self.delta_ffn = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, dim),
        )
        # gating: token-level
        self.gate_tok = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, 1),
        )
        self.out = nn.Linear(dim, out_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)
        nn.init.xavier_uniform_(self.gate_tok[0].weight)
        nn.init.zeros_(self.gate_tok[0].bias)

        nn.init.zeros_(self.gate_tok[2].weight)        # shape [1, dim]
        nn.init.constant_(self.gate_tok[2].bias, -5.0) # sigmoid(-5)≈0.0067
  
    def forward(self, H_t, H_m, mask_t=None, mask_m=None, normalize=True):
        # cross-attn
        mask_mr = (mask_m==0)
        T = self.ln_t(H_t)                       # [B, Lt, D]
        M = self.ln_m(H_m)                       # [B, Lm, D]
        delta_t, _ = self.t2m(T, M, M, key_padding_mask=mask_mr)   # text attends math -> [B,Lt,D]
        delta_t = self.delta_ffn(self.ln_delta(delta_t)) 
        g = torch.sigmoid(self.gate_tok(delta_t))               # [B,L,D]->[B,L,1]
        # residual connection：Concat-MLP + Gated tensor
        T_fused = T + g * delta_t   # [B, Lt, D]
        Z = self.out(T_fused)
        out_mask = mask_t
        return Z, out_mask
    

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
