from torch import Tensor
from typing import Optional

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
        postprocess: str,
        criterion: nn.Module,
        embs: Tensor,
        attn_mask: Tensor,
        n_exprs: Optional[int],
) -> float:
    print("postprocess is:", postprocess)
    if postprocess == "cls":
        embs = embs[:, 0, :].view(-1, n_exprs, embs.size(dim=-1))

    elif postprocess in {"mean", "max", "maxsim"}:
        # sep_ids = attn_mask.int().sum(dim=-1) - 1
        # batch_ids = torch.arange(
        #     start=0,
        #     end=attn_mask.size(dim=0),
        #     dtype=torch.int64,
        #     device=attn_mask.device,
        # )
        # attn_mask[batch_ids, sep_ids] = False
        # attn_mask[:, 0] = False

        if postprocess == "mean":
            # embs[attn_mask == 0] = 0.0
            embs = embs.sum(dim=-2, keepdim=False)
            n_tokens = attn_mask.int().sum(dim=1, keepdim=False) \
                .float().unsqueeze(dim=-1)
            embs = embs / n_tokens
        elif postprocess == "max":
            embs[~attn_mask] = float("-inf")
            embs = embs.max(dim=-2, keepdim=False).values

        elif postprocess == "maxsim":
            _, L, D = embs.size()
            print(embs.shape)
            embs = embs.view(-1, n_exprs, L, D)
            attn_mask = attn_mask.view(-1, n_exprs, L)
            query = embs[:, 0, :, :]
            pos_key = embs[:, 1, :, :]
            neg_key = embs[:, 2:, :, :]
            query_mask = attn_mask[:, 0, :]
            pos_mask = attn_mask[:, 1, :]
            neg_mask = attn_mask[:, 2:, :]

            return criterion(
                query=query,
                pos_key=pos_key,
                neg_key=neg_key,
                query_mask=query_mask,
                pos_mask=pos_mask,
                neg_mask=neg_mask,
            )

    embs = embs.view(-1, n_exprs, embs.size(dim=-1))
    query = embs[:, 0, :]
    pos_key = embs[:, 1, :]
    neg_key = embs[:, 2:, :]

    return criterion(query=query, pos_key=pos_key, neg_key=neg_key)

from itertools import islice

def train_epoch(
        model: nn.Module,
        ckpt_last: str,
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
    model.train(mode=True)

    loader_tqdm = tqdm(iterable=dataloader, position=1, leave=False)
    loader_tqdm.set_description(desc=f"[{timestamp()}] [Batch 0]", refresh=True)
    n_iters = len(dataloader)
    name = model.__class__.__name__.lower()
    name = "math_enc_old"
    loss_meter = AverageMeter()

    for i, batch in enumerate(iterable=loader_tqdm):
        if i < init_batch:
            continue
        print(i)
        token_ids = batch["input_ids"].to(device=device)
        attn_mask = batch["attention_mask"].to(device=device)

        optimizer.zero_grad()

        if name == "dualenc":
            raise NotImplementedError
        elif name == "mathenc":
            embs = model(token_ids=token_ids, attn_mask=attn_mask, cache_pos=None)
        elif name == "bert":
            query_ids = torch.arange(
                start=0, end=token_ids.size(dim=0), step=n_exprs
            )
            query = token_ids[query_ids]
            # change from [PAD] (0) to [MASK] (103)
            query[query == 0] = 103
            token_ids[query_ids] = query
            query_mask = attn_mask[query_ids]
            query_mask[query_mask == 0] = 1
            attn_mask[query_ids] = query_mask
            embs = model(token_ids=token_ids, attn_mask=attn_mask)
            punct_mask = batch["punct_mask"].to(device=device)
            query_mask = punct_mask[query_ids]
            query_mask[query_mask == 0] = 1
            punct_mask[query_ids] = query_mask
            attn_mask = attn_mask & punct_mask
        elif name == "math_enc_old":
            embs = model(tokens=token_ids, mask=attn_mask, cache_pos=None)
            attn_mask = attn_mask.squeeze(dim=(-3,-2))  # [B, 1, 1, L] -> [B, L]
            n_pad = attn_mask.int().sum(dim=-1)
            eoe_ids = attn_mask.size(dim=-1)-n_pad-1
            batch_ids = torch.arange(
                start=0, end=attn_mask.size(dim=0), dtype=torch.int64, device=attn_mask.device
            )
            attn_mask[batch_ids, eoe_ids] = True
            attn_mask[:, 0] = True
            attn_mask = (~attn_mask).long() #convert transformer mask to bert mask
            print(attn_mask)
        else:
            raise ValueError(f"Invalid model class `{name}`")

        loss = compute_loss(
            postprocess=postprocess,
            criterion=criterion,
            embs=embs,
            attn_mask=attn_mask,
            n_exprs=n_exprs,
        )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
        lr_scheduler.step_update(n_iters * epoch + i)

        loss_meter.update(loss.item(), n=token_ids.size(dim=0))
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
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "batch": i,
                    "loss": loss,
                },
                ckpt_last,
            )
            loader_tqdm.write(
                s=f"[{timestamp()}] [Epoch {epoch}] [Batch {i}] Saved model to "
                  f"`{ckpt_last}`"
            )

    return loss_meter.avg


def train_model(
        model: nn.Module,
        ckpt_last: str,
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
    torch.autograd.set_detect_anomaly(True)
    path, _ = os.path.split(p=ckpt_last)
    if not os.path.exists(path=path):
        os.makedirs(name=path, exist_ok=True)

    model.to(device=device)

    params = train_params(model=model)
    log_info(f"Total trainable parameters {params * 1e-6:.4f}M")

    init_epoch = 0
    init_batch = 0
    best_loss = float('inf')

    if os.path.exists(path=ckpt_last):
        ckpt = torch.load(f=ckpt_last, map_location=device)
        model.load_state_dict(state_dict=ckpt["model_state_dict"])
        optimizer.load_state_dict(state_dict=ckpt["optimizer_state_dict"])
        lr_scheduler.load_state_dict(state_dict=ckpt["lr_scheduler_state_dict"])
        init_batch = ckpt["batch"]+1
        print("init batch is:", init_batch)
        init_epoch = ckpt["epoch"]+1 if init_batch == 0 else ckpt["epoch"]
        print("loss:", ckpt["loss"])
        filename = os.path.basename(p=ckpt_last)
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
            model=model,
            ckpt_last=ckpt_last,
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
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state": lr_scheduler.state_dict(),
                "epoch": epoch,
                "batch": -1,
                "loss": loss,
            },
            ckpt_last,
        )
        epoch_tqdm.write(
            s=f"[{timestamp()}] [Epoch {epoch}]: Saved best model to "
                f"`{ckpt_last}`"
        )
