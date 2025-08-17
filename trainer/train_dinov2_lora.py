#!/usr/bin/env python3
import os, json, math, time, random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# local imports
from config.opts import parse_args, add_lora_args
from data.dataset import NucleusCSV
from utils.utils import (
    set_seed, make_weighted_sampler, compute_metrics,
    init_metrics_logger, log_metrics
)
from model.lora import apply_lora_to_timm_vit


# -----------------------------
# DDP helpers
# -----------------------------
def setup_ddp():
    """Initialize process group if torchrun provided env vars. Returns (is_ddp, local_rank, world_size, global_rank)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return True, local_rank, world_size, global_rank
    return False, 0, 1, 0

def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0

def ddp_allreduce_scalar(val: float, device) -> float:
    """Average a scalar across ranks."""
    t = torch.tensor([val], dtype=torch.float32, device=device)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return float(t.item())

def ddp_allreduce_counts(counts: np.ndarray, device) -> np.ndarray:
    """Sum an integer count vector across ranks."""
    t = torch.tensor(counts, dtype=torch.int64, device=device)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.cpu().numpy()


# -----------------------------
# Val prediction histogram (DDP aware)
# -----------------------------
def per_class_counts_ddp(val_loader, model, device, num_classes):
    model.eval()
    local = np.zeros(num_classes, dtype=np.int64)
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        for x, _ in val_loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            preds = logits.argmax(1).cpu().numpy()
            for p in preds:
                local[p] += 1
    return local  # caller will allreduce


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args(add_lora_args)
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- DDP init ----
    is_ddp, local_rank, world_size, global_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process():
        print("==> Launch config")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")
        print(f"DDP: {is_ddp} | world_size={world_size} | local_rank={local_rank} | global_rank={global_rank}")

    # ---- Load class map first (we need num_classes BEFORE building model) ----
    with open(args.class_map, "r") as f:
        class_to_idx = json.load(f)
    classes = sorted(class_to_idx, key=lambda k: class_to_idx[k])
    num_classes = len(classes)
    if is_main_process():
        print(f"Classes ({num_classes}): {classes}")

    # ---- Build model (timm DINOv2 ViT by default) ----
    model = timm.create_model(args.model, pretrained=True, num_classes=num_classes)

    # Get normalization from model cfg (fallback to ImageNet defaults)
    mean = tuple(model.pretrained_cfg.get('mean', IMAGENET_DEFAULT_MEAN))
    std  = tuple(model.pretrained_cfg.get('std',  IMAGENET_DEFAULT_STD))

    # Freeze everything then unfreeze the classifier head
    for p in model.parameters():
        p.requires_grad = False
    # timm ViTs expose .head for the classifier
    for p in model.head.parameters():
        p.requires_grad = True

    # Inject LoRA adapters into last N blocks
    lora_params = apply_lora_to_timm_vit(
        model,
        last_n_blocks=args.lora_blocks,
        r=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    model.to(device)

    # Wrap with DDP after moving to device
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)

    # ---- Datasets ----
    train_ds = NucleusCSV(
        args.train_csv, class_to_idx,mean=mean, std=std,
        use_img_uniform=args.use_img_uniform,
        augment=True
    )
    val_ds = NucleusCSV(
        args.val_csv, class_to_idx, mean=mean, std=std,
        use_img_uniform=args.use_img_uniform,
        augment=False
    )

    if is_main_process():
        print(f"Train samples: {len(train_ds)}")
        print(f"Val samples:   {len(val_ds)}")

    # ---- Samplers & Loaders ----
    # If DDP, always use DistributedSampler. If single GPU and args.sampler == "weighted", use WeightedRandomSampler.
    pw = args.num_workers > 0

    if is_ddp:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=global_rank, shuffle=True, drop_last=False)
        val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=global_rank, shuffle=False, drop_last=False)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=train_sampler,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=pw, prefetch_factor=4 if pw else None
        )
    else:
        if args.sampler == "weighted":
            # Build weights from the filtered dataframe inside the dataset
            tdf = train_ds.df
            sampler = make_weighted_sampler(tdf, "cell_type")
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, sampler=sampler,
                num_workers=args.num_workers, pin_memory=True,
                persistent_workers=pw, prefetch_factor=4 if pw else None
            )
        else:
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True,
                persistent_workers=pw, prefetch_factor=4 if pw else None
            )
        val_sampler = None  # single GPU
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
        num_workers=max(1, args.num_workers // 2), pin_memory=True,
        persistent_workers=pw
    )

    # ---- Loss ----
    # If we aren't using a WeightedRandomSampler (i.e., DDP or plain shuffle),
    # use class-weighted CE based on the training distribution to mitigate imbalance.
    cls_weights = None
    use_class_weights = (is_ddp or args.sampler != "weighted")
    if use_class_weights:
        counts = train_ds.df["cell_type"].value_counts()
        weights = np.zeros(len(classes), dtype=np.float32)
        for i, c in enumerate(classes):
            weights[i] = 1.0 / max(1, counts.get(c, 0))
        # normalize to sum=1 is optional; CE only needs relative scaling
        cls_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=cls_weights)

    # ---- Optimizer & Schedule ----
    # Two groups: LoRA params vs head params
    # NOTE: if model is DDP, need to access .module to count head params (already frozen/unfrozen)
    raw_model = model.module if isinstance(model, DDP) else model
    head_params = [p for p in raw_model.head.parameters() if p.requires_grad]

    if is_main_process():
        lora_count = sum(p.numel() for p in lora_params)
        head_count = sum(p.numel() for p in head_params)
        print(f"Trainable (LoRA): {lora_count:,}")
        print(f"Trainable (head): {head_count:,}")

    optim_groups = [
        {"params": lora_params, "lr": args.lr_lora, "weight_decay": args.weight_decay},
        {"params": head_params, "lr": args.lr_head, "weight_decay": args.weight_decay},
    ]
    optimizer = torch.optim.AdamW(optim_groups)

    if args.cosine:
        total_epochs = args.epochs
        warmup = max(0, min(args.warmup_epochs, total_epochs - 1))

        def lr_lambda(epoch):
            if epoch < warmup:
                return float(epoch + 1) / float(max(1, warmup))
            # cosine from warmup -> total_epochs
            progress = (epoch - warmup) / float(max(1, total_epochs - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # Save the used class map (rank 0)
    if is_main_process():
        with open(Path(args.out_dir, "class_to_idx.used.json"), "w") as f:
            json.dump(class_to_idx, f, indent=2)

    best_acc = 0.0
    best_path = str(Path(args.out_dir, "best_dinov2_lora.pt"))

    # Metrics logger (rank 0 only)
    if is_main_process():
        run_id = time.strftime("%Y%m%d-%H%M%S")
        metrics_f, metrics_w = init_metrics_logger(args.out_dir, filename=f"metrics_{run_id}.csv")
    else:
        run_id, metrics_f, metrics_w = "", None, None

    if is_main_process():
        print("\nStarting training...")
        print(f"LoRA on last {args.lora_blocks} blocks | rank={args.lora_rank} alpha={args.lora_alpha} dropout={args.lora_dropout}")
        print(f"LRs: head={args.lr_head}  lora={args.lr_lora}  | cosine={args.cosine} warmup={args.warmup_epochs}\n")

    # ---- Train loop ----
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        if is_ddp:
            # ensure each rank shuffles differently
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

        raw_model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()

            # gradient clipping
            scaler.unscale_(optimizer)
            params_to_clip = []
            for g in optimizer.param_groups:
                params_to_clip += [p for p in g["params"] if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * y.size(0)
            c, t = compute_metrics(logits.detach(), y)
            train_correct += c
            train_total += t

        # Reduce train aggregates across ranks
        if is_ddp:
            tt = torch.tensor([train_loss_sum, train_correct, train_total], dtype=torch.float64, device=device)
            dist.all_reduce(tt, op=dist.ReduceOp.SUM)
            train_loss_sum, train_correct, train_total = float(tt[0].item()), int(tt[1].item()), int(tt[2].item())

        if scheduler is not None:
            scheduler.step()

        # ---- Validation ----
        raw_model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0

        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss_sum += loss.item() * y.size(0)
                c, t = compute_metrics(logits, y)
                val_correct += c
                val_total += t

        # Reduce val aggregates
        if is_ddp:
            vv = torch.tensor([val_loss_sum, val_correct, val_total], dtype=torch.float64, device=device)
            dist.all_reduce(vv, op=dist.ReduceOp.SUM)
            val_loss_sum, val_correct, val_total = float(vv[0].item()), int(vv[1].item()), int(vv[2].item())

        train_loss = train_loss_sum / max(1, train_total)
        train_acc  = train_correct / max(1, train_total)
        val_loss   = val_loss_sum   / max(1, val_total)
        val_acc    = val_correct    / max(1, val_total)
        secs = time.time() - t0

        if is_main_process():
            print(f"Epoch {epoch:02d}/{args.epochs}  |  "
                  f"train_loss {train_loss:.4f}  train_acc {train_acc:.3f}  "
                  f"val_loss {val_loss:.4f}  val_acc {val_acc:.3f}  |  {secs:.1f}s")

            # Log metrics CSV
            lr_lora = optimizer.param_groups[0]["lr"]
            lr_head = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else lr_lora
            log_metrics(metrics_w, run_id, epoch, train_loss, train_acc, val_loss, val_acc, lr_head, lr_lora, secs)
            metrics_f.flush()

            # Optional: prediction breakdown every 5 epochs (DDP-aware)
            if epoch == 1 or epoch % 5 == 0:
                local_counts = per_class_counts_ddp(val_loader, model, device, num_classes)
                counts = ddp_allreduce_counts(local_counts, device)
                total_preds = counts.sum()
                print("  Val prediction breakdown (all ranks):")
                for i, cls in enumerate(classes):
                    frac = (counts[i] / max(1, total_preds)) * 100.0
                    print(f"    {cls}: {counts[i]} ({frac:.1f}%)")

            # Save best
            if val_acc > best_acc:
                best_acc = val_acc
                to_save = raw_model.state_dict()
                torch.save({
                    "model": to_save,
                    "args": vars(args),
                    "classes": classes,
                    "val_acc": best_acc,
                }, best_path)
                print(f"  ✓ Saved new best to {best_path}")

    if is_main_process():
        if 'metrics_f' in locals() and metrics_f is not None:
            metrics_f.close()
        print(f"\nDone. Best val_acc={best_acc:.3f}")
        print(f"Checkpoint: {best_path}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
