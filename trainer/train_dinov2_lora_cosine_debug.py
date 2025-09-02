#!/usr/bin/env python3
import os, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from torch.utils.data import DataLoader
import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from sklearn.metrics import f1_score

# local imports
from config.opts import parse_args, add_lora_args
from data.dataset import NucleusCSV
from utils.utils import (
    set_seed, make_weighted_sampler,
    init_metrics_logger, log_metrics,
    compute_per_class_metrics, print_per_class_summary,
    class_counts_from_df, maybe_init_wandb
)
from model.lora import apply_lora_to_timm_vit


# -----------------------------
# Cosine head with fixed scale
# -----------------------------
class CosineHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, s_init: float = 30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")
        # freeze scale
        self._s_unconstrained = nn.Parameter(
            torch.log(torch.expm1(torch.tensor(s_init, dtype=torch.float32))),
            requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        s = F.softplus(self._s_unconstrained) + 1e-6
        return s * (x @ w.t())


def main():
    args = parse_args(add_lora_args)

    print("Training DINOv2 + LoRA (DEBUG CONFIG, plain CE)")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load classes ----
    with open(args.class_map, "r") as f:
        class_to_idx = json.load(f)
    classes = sorted(class_to_idx, key=lambda k: class_to_idx[k])
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")

    # ---- Build model ----
    model = timm.create_model(
        args.model, pretrained=True,
        num_classes=num_classes, img_size=args.img_size
    )
    mean = tuple(model.pretrained_cfg.get('mean', IMAGENET_DEFAULT_MEAN))
    std  = tuple(model.pretrained_cfg.get('std',  IMAGENET_DEFAULT_STD))

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Replace head
    in_dim = getattr(model.head, "in_features", getattr(model, "num_features"))
    model.head = CosineHead(in_dim, num_classes, s_init=30.0)
    for p in model.head.parameters():
        p.requires_grad = True

    # Inject LoRA
    lora_params = apply_lora_to_timm_vit(
        model,
        last_n_blocks=args.lora_blocks,
        r=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    # Unfreeze norms
    total_blocks = len(model.blocks)
    start = max(0, total_blocks - args.lora_blocks)
    norms = []
    for i in range(start, total_blocks):
        for name in ("norm1", "norm2"):
            m = getattr(model.blocks[i], name, None)
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = True
                norms += list(m.parameters())

    model.to(device)

    # ---- Datasets ----
    train_ds = NucleusCSV(
        args.train_csv, class_to_idx, size=args.img_size, mean=mean, std=std,
        use_img_uniform=args.use_img_uniform, augment=True
    )
    val_ds = NucleusCSV(
        args.val_csv, class_to_idx, size=args.img_size, mean=mean, std=std,
        use_img_uniform=args.use_img_uniform, augment=False
    )
    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    counts_np = class_counts_from_df(train_ds.df, label_col="cell_type", classes=classes)
    print("Train class distribution:")
    for cls, c in zip(classes, counts_np):
        print(f"  {cls:>24}: {c}")

    # ---- DataLoaders ----
    sampler = make_weighted_sampler(train_ds.df, label_col="cell_type")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

  

    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16))

    # ---- Optimizer ----
    optim_groups = [
        {"params": lora_params, "lr": args.lr_lora, "weight_decay": args.weight_decay},
        {"params": model.head.parameters(), "lr": args.lr_head, "weight_decay": args.weight_decay},
        {"params": norms, "lr": args.lr_head, "weight_decay": args.weight_decay},
    ]
    optimizer = torch.optim.AdamW(optim_groups)
    
    """
    # One‑batch overfit sanity check
    xb, yb = next(iter(train_loader))
    xb, yb = xb.to(device), yb.to(device)
    model.train()
    steps = 200
    print(f"[DEBUG] One‑batch overfit for {steps} steps on batch size {xb.size(0)}")
    for t in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=torch.cuda.is_available()):
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)  # plain CE, no logit adjustment
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if (t + 1) % 20 == 0:
            acc = (logits.argmax(1) == yb).float().mean().item()
            print(f"  step {t+1:4d}  loss {loss.item():.4f}  acc {acc:.3f}")
    return
    #------------------
    """

    best_macro_f1 = -1.0
    best_path = str(Path(args.out_dir, "debug_best.pt"))

    run_id = time.strftime("%Y%m%d-%H%M%S")
    metrics_f, metrics_w = init_metrics_logger(args.out_dir, filename=f"metrics_debug_{run_id}.csv", classes=classes)

    print("\nStarting debug training (plain CE, frozen scale, no LA)...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_sum, train_correct, train_total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=torch.cuda.is_available()):
                logits = model(x)
                loss = F.cross_entropy(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * y.size(0)
            preds = logits.argmax(1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)

        # ---- Validation ----
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        all_preds, all_targets = [], []
        with torch.inference_mode(), torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=torch.cuda.is_available()):
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                val_loss_sum += loss.item() * y.size(0)
                preds = logits.argmax(1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())

        # metrics
        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total
        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        all_preds = torch.cat(all_preds) if all_preds else torch.empty(0)
        all_targets = torch.cat(all_targets) if all_targets else torch.empty(0)

        macro_f1 = 0.0
        if all_preds.numel() > 0:
            macro_f1 = f1_score(all_targets.numpy(), all_preds.numpy(), average="macro", zero_division=0)

        secs = time.time() - t0
        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss {train_loss:.4f} train_acc {train_acc:.3f} | "
              f"val_loss {val_loss:.4f} val_acc {val_acc:.3f} | "
              f"macro_f1 {macro_f1:.3f} | {secs:.1f}s")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)
            print(f"  ✓ Saved new best checkpoint (macro F1={best_macro_f1:.3f})")

        log_metrics(metrics_w, run_id, epoch, train_loss, train_acc, val_loss, val_acc,
                    args.lr_head, args.lr_lora, secs, None, classes)
        metrics_f.flush()

    if metrics_f is not None:
        metrics_f.close()
    print(f"\nDone. Best macro F1={best_macro_f1:.3f} | Checkpoint: {best_path}")


if __name__ == "__main__":
    main()
