from torch import Tensor
from typing import Optional
import logger
import os
import torch
import torch.nn as nn
import torch.optim as optim
from avg_meter import AverageMeter
from logger import log_info, timestamp
from timm.scheduler.scheduler import Scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import train_params


def compute_loss(
        criterion: nn.Module,
        text_embs: Tensor,
        text_attn_mask: Tensor,
        formula_embs: Tensor,
        formula_attn_mask: Tensor,
        model_fuse: nn.Module,
        n_exprs: Optional[int],
) -> float:
    _, tL, tD = text_embs.size()
    _, fL, fD = formula_embs.size()
    assert text_embs.shape[0]==formula_embs.shape[0]
    
    #find q, pos, neg of text
    text_embs = text_embs.view(-1, n_exprs, tL, tD)
    text_attn_mask = text_attn_mask.view(-1, n_exprs, tL)
    text_q = text_embs[:, 0, :, :]
    text_pos = text_embs[:, 1, :, :]
    text_neg = text_embs[:, 2, :, :]
    text_q_mask = text_attn_mask[:, 0, :]
    text_pos_mask = text_attn_mask[:, 1, :]
    text_neg_mask = text_attn_mask[:, 2, :]
    
    #find q, pos, neg of formula
    formula_embs = formula_embs.view(-1, n_exprs, fL, fD)
    formula_attn_mask = formula_attn_mask.view(-1, n_exprs, fL)
    formula_q = formula_embs[:, 0, :, :]
    formula_pos = formula_embs[:, 1, :, :]
    formula_neg = formula_embs[:, 2, :, :]
    formula_q_mask = formula_attn_mask[:, 0, :]
    formula_pos_mask = formula_attn_mask[:, 1, :]
    formula_neg_mask = formula_attn_mask[:, 2, :]
    print(text_q.dtype)
    print(formula_q.dtype)
    fused_q = model_fuse(H_t=text_q, H_m=formula_q, mask_t=text_q_mask, mask_m=formula_q_mask, normalize=True)
    mask_q = torch.cat((text_q_mask, formula_q_mask), dim=1)
    assert fused_q.size(1) == mask_q.size(1)
    # print("text_q mask:", text_q)
    # print("formula_pos mask shape:", formula_pos.shape)
    assert fused_q.shape[0] == mask_q.shape[0]
    fused_pos = model_fuse(H_t=text_pos, H_m=formula_pos, mask_t=text_pos_mask, mask_m=formula_pos_mask, normalize=True)
    print("text_pos mask shape:", text_pos.shape)
    print("formula_pos mask shape:", formula_pos.shape)
    mask_pos = torch.cat((text_pos_mask, formula_pos_mask), dim=1)
    assert fused_pos.size(1) == mask_pos.size(1)
    # print("cat mask:", mask_pos)
    # print("cat mask shape:", mask_pos.shape)
    assert fused_pos.shape[0] == mask_pos.shape[0]
    # print("text_neg shape:", text_neg.shape)
    # print("formula_neg shape:", formula_neg.shape)
    # print("formula_neg_mask shape", formula_neg_mask.shape)
    fused_neg = model_fuse(H_t=text_neg, H_m=formula_neg, mask_t=text_neg_mask, mask_m=formula_neg_mask, normalize=True)
    mask_neg = torch.cat((text_neg_mask, formula_neg_mask), dim=1)
    assert fused_neg.size(1) == mask_neg.size(1)
    assert fused_neg.shape[0] == mask_neg.shape[0]
    return criterion(
        query=fused_q,
        pos_key=fused_pos,
        neg_key=fused_neg,
        query_mask=mask_q,
        pos_mask=mask_pos,
        neg_mask=mask_neg,
    )

from itertools import islice

def train_epoch(
        model_text: nn.Module,
        model_formula: nn.Module,
        model_fuse: nn.Module,
        ckpt_fuse: str,
        optimizer: optim.Optimizer,
        lr_scheduler: Scheduler,
        postprocess: str,
        n_exprs: int,
        criterion: nn.Module,
        max_norm: float,
        device: torch.device,
        dataloader: DataLoader,
        epoch: int,
        init_batch: int,
        save_every_n_iters: int,
) -> float:
    model_fuse.train(mode=True)

    loader_tqdm = tqdm(iterable=dataloader, position=1, leave=False)
    loader_tqdm.set_description(desc=f"[{timestamp()}] [Batch 0]", refresh=True)
    n_iters = len(dataloader)
    
    loss_meter = AverageMeter()

    for i, batch in enumerate(iterable=loader_tqdm):
        if i < init_batch:
            continue
        text_token_ids = batch["text"]["input_ids"].to(device=device)
        text_attn_mask = batch["text"]["attention_mask"].to(device=device)
        formula_token_ids = batch["formula"]["src"].to(device=device)
        formula_attn_mask = batch["formula"]["src_mask"].to(device=device)

        optimizer.zero_grad()
        query_ids = torch.arange(
            start=0, end=text_token_ids.size(dim=0), step=n_exprs
        )
        query = text_token_ids[query_ids]
        # change from [PAD] (0) to [MASK] (103)
        query[query == 0] = 103
        text_token_ids[query_ids] = query
        query_mask = text_attn_mask[query_ids]
        query_mask[query_mask == 0] = 1
        text_attn_mask[query_ids] = query_mask
        with torch.no_grad():
            text_embs = model_text(token_ids=text_token_ids, attn_mask=text_attn_mask)
        # print( text_embs.dtype)

        punct_mask = batch["text"]["punct_mask"].to(device=device)
        query_mask = punct_mask[query_ids]
        query_mask[query_mask == 0] = 1
        punct_mask[query_ids] = query_mask
        text_attn_mask = text_attn_mask & punct_mask
        with torch.no_grad():
            formula_embs = model_formula(tokens=formula_token_ids, mask=formula_attn_mask, cache_pos=None)
        print(formula_embs.dtype)
        formula_attn_mask = formula_attn_mask.squeeze(dim=(-3, -2)) # [B, 1, 1, L] -> [B, L]
        n_pad = formula_attn_mask.int().sum(dim=-1)
        eoe_ids = formula_attn_mask.size(dim=-1) - n_pad - 1
        batch_ids = torch.arange(
            start=0, end=formula_attn_mask.size(dim=0), dtype=torch.int64, device=formula_attn_mask.device
        )
        formula_attn_mask[batch_ids, eoe_ids] = True
        formula_attn_mask[:, 0] = True
        formula_embs[formula_attn_mask] = 0 # pad is true in this mask
        formula_attn_mask = (~formula_attn_mask).long()    # inverse valid token and pad, valid token is true now
        print("new formula mask", formula_attn_mask)
        # fused_emb = model_fuse(H_t=text_embs, H_m=formula_embs, mask_t=text_attn_mask, mask_m=formula_attn_mask, normalize=True)

        loss = compute_loss(
            criterion=criterion,
            text_embs=text_embs,
            text_attn_mask=text_attn_mask,
            formula_embs=formula_embs,
            formula_attn_mask=formula_attn_mask,
            model_fuse=model_fuse,
            n_exprs=n_exprs,
        )

        loss.backward()
        nn.utils.clip_grad_norm_(model_fuse.parameters(), max_norm=max_norm)
        optimizer.step()
        lr_scheduler.step_update(n_iters * epoch + i)

        loss_meter.update(loss.item(), n=formula_token_ids.size(dim=0))
        loader_tqdm.set_description(
            desc=f"[{timestamp()}] [Batch {i+1}]: "
                 f"train loss {loss_meter.avg:.6f}",
            refresh=True,
        )

        if (i + 1) % save_every_n_iters == 0:
            n_steps = n_iters * epoch + (i+1)
            for param_group in optimizer.param_groups:
                loader_tqdm.write(f"[{timestamp()}] [Step {n_steps}] Current LR {param_group['lr']:.8f}")

            torch.save(
                {
                    "model_state_dict": model_fuse.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "batch": i,
                    "loss": loss,
                },
                ckpt_fuse,
            )
            loader_tqdm.write(
                s=f"[{timestamp()}] [Epoch {epoch}] [Batch {i}] Saved model to "
                  f"`{ckpt_fuse}`"
            )

    return loss_meter.avg


def train_model(
        model_text: nn.Module,
        ckpt_text: str,
        model_formula: nn.Module,
        ckpt_formula: str,
        model_fuse: nn.Module,
        ckpt_fuse: str,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler.LRScheduler,
        postprocess: str,
        n_exprs: int,
        criterion: nn.Module,
        max_norm: float,
        device: torch.device,
        n_epochs: int,
        dataloader: DataLoader,
        save_every_n_iters: int,
) -> None:
    path, _ = os.path.split(p=ckpt_fuse)
    if not os.path.exists(path=path):
        os.makedirs(name=path, exist_ok=True)

    #load models
    model_text.to(device=device)
    ckpt_text_model = torch.load(f=ckpt_text, map_location=device)
    missing_keys, unexpected_keys = model_text.load_state_dict(state_dict=ckpt_text_model, strict=False)
    model_text.eval()
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    logger.log_info(f"Loaded model '{ckpt_text}'")
    # params = train_params(model=model_text)
    # log_info(f"Total trainable parameters {params * 1e-6:.4f}M")
    model_formula.to(device=device)
    ckpt_formula_old = torch.load(f=ckpt_formula, map_location=device)
    missing_keys, unexpected_keys = model_formula.load_state_dict(state_dict=ckpt_formula_old["model_state"], strict=False)
    model_formula.eval()
    print(f"Missing keys in formula ckpt: {missing_keys}")
    print(f"Unexpected keys in formula ckpt: {unexpected_keys}")
    logger.log_info(f"Loaded model '{ckpt_formula}'")

    model_fuse.to(device=device)
    params = train_params(model=model_fuse)
    log_info(f"Total trainable parameters {params * 1e-6:.4f}M")

    init_epoch = 0
    init_batch = 0

    if os.path.exists(path=ckpt_fuse):
        ckpt_path = ckpt_fuse
        ckpt_fuse = torch.load(f=ckpt_fuse, map_location=device)
        model_fuse.load_state_dict(state_dict=ckpt_fuse["model_state_dict"])
        optimizer.load_state_dict(state_dict=ckpt_fuse["optimizer_state_dict"])
        lr_scheduler.load_state_dict(state_dict=ckpt_fuse["lr_scheduler_state_dict"])
        init_batch = ckpt_fuse["batch"]+1
        print("init batch is:", init_batch)
        init_epoch = ckpt_fuse["epoch"]+1 if init_batch == 0 else ckpt_fuse["epoch"]
        print("loss:", ckpt_fuse["loss"])
        filename = os.path.basename(p=ckpt_path)
        log_info(f"Loaded `{filename}`")

    epoch_tqdm = tqdm(
        iterable=range(init_epoch, n_epochs),
        desc=f"[{timestamp()}] [Epoch {init_epoch}]",
        position=0,
        leave=True,
    )

    for epoch in epoch_tqdm:
        epoch_tqdm.set_description(
            desc=f"[{timestamp()}] [Epoch {epoch}]",
            refresh=True,
        )
        loss = train_epoch(
            model_text=model_text,
            model_formula=model_formula,
            model_fuse=model_fuse,
            ckpt_fuse=ckpt_fuse,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            postprocess=postprocess,
            n_exprs=n_exprs,
            criterion=criterion,
            max_norm=max_norm,
            device=device,
            dataloader=dataloader,
            epoch=epoch,
            init_batch=init_batch,
            save_every_n_iters=save_every_n_iters,
        )

        init_batch = 0

        epoch_tqdm.write(s=f"[{timestamp()}] [Epoch {epoch}] loss {loss:.6f}")

        torch.save(
            {
                "model_state_dict": model_text.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state": lr_scheduler.state_dict(),
                "epoch": epoch,
                "batch": -1,
                "loss": loss,
            },
            ckpt_fuse,
        )
        epoch_tqdm.write(
            s=f"[{timestamp()}] [Epoch {epoch}]: Saved best model to "
                f"`{ckpt_fuse}`"
        )
