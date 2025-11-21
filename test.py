import torch
e_t = torch.tensor([[1,11,2,22,3,0,0,0], [1,11,2,22,3,0,0,0]])
e_m = torch.tensor([[1,2,3,4,0,0], [1,2,3,4,0,0]])
mask_t = torch.tensor([[1,0,1,0,1,0,0,0], [1,0,1,0,1,0,0,0]])
mask_m = torch.tensor([[0,1,1,0,0,0], [0,1,0,0,0,0]])
def compact_by_mask(e_t, e_m, mask_t, mask_m):
    # e_t: [B,Lt,D], e_m: [B,Lm,D], mask_*: [B,L*], 1/True=valid
    # B, _, D = e_t.shape
    fused = torch.cat([e_t, e_m], dim=1)                      # [B,Lt+Lm,D]
    fused_mask = torch.cat([mask_t, mask_m], dim=1).bool()    # [B,Lt+Lm]

    # 构造“稳定排序”的 key：先按 pad(=1) / valid(=0)，再按原始位置
    key1 = (~fused_mask).to(torch.int64)                      # valid→0, pad→1
    order = torch.arange(fused.size(1), device=fused.device).unsqueeze(0).expand_as(key1)
    scores = key1 * (fused.size(1) + 1) + order               # [B,L], 两关键字合一
    print(scores)
    idx = torch.argsort(scores, dim=1) 
    print(idx)                       # 稳定地把 valid 放前
    # idx = idx.unsqueeze(-1)                # [B,L, D]
    print(idx.shape)
    fused_compact = fused.gather(1, idx)   
    print('fused_mask', fused_mask)                   # [B,Lt+Lm,D]
    mask_compact  = fused_mask.gather(1, idx.squeeze(-1))     # [B,Lt+Lm]
    print('mask_compact', mask_compact)
    # （可选）裁到每条样本的有效长度上限，减少 pad
    L = mask_compact.sum(1)                                   # [B]
    Lmax = int(L.max().item())
    fused_compact = fused_compact[:, :Lmax]
    mask_compact  = mask_compact[:, :Lmax]
    mask_compact = mask_compact.long()
    return fused_compact, mask_compact
fused_compact, mask_compact = compact_by_mask(e_t, e_m, mask_t, mask_m)
print(fused_compact)
print(mask_compact)