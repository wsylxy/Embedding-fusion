from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from criterions.registry import register_criterion


class InfoNCE(nn.Module):
    def __init__(self, temperature: float, reduction: str) -> None:
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
            self,
            query: Tensor,
            pos_key: Tensor,
            neg_key: Tensor,
    ) -> Tensor:
        # [B, D] -> [B, D]
        query = F.normalize(input=query, p=2.0, dim=-1, eps=1e-12)
        # [B, D] -> [B, D]
        pos_key = F.normalize(input=pos_key, p=2.0, dim=-1, eps=1e-12)
        # [B, NG, D] -> [B, NG, D]
        neg_key = F.normalize(input=neg_key, p=2.0, dim=-1, eps=1e-12)

        # [B, D] * [B, D] -> [B, 1]
        pos_logit = torch.sum(query*pos_key, dim=1, keepdim=True)

        # [B, D] -> [B, 1, D]
        query = query.unsqueeze(dim=1)
        # [B, 1, D] @ [B, D, NG] -> [B, 1, NG]
        neg_logit = query @ neg_key.transpose(dim0=-2, dim1=-1)
        # [B, 1, NG] -> [B, NG]
        neg_logit = neg_logit.squeeze(dim=1)

        # [B, 1] [B, NG] -> [B, 1 + NG]
        logits = torch.cat(tensors=[pos_logit, neg_logit], dim=1)
        labels = torch.zeros(
            logits.size(dim=0),
            dtype=torch.int64,
            device=query.device,
        )
        loss = F.cross_entropy(
            input=logits/self.temperature,
            target=labels,
            reduction=self.reduction,
        )

        return loss


@register_criterion(name="infonce")
def build_model(cfg) -> nn.Module:
    return InfoNCE(
        temperature=cfg.CRITERION.INFONCE.TEMPERATURE,
        reduction=cfg.CRITERION.INFONCE.REDUCTION,
    )
