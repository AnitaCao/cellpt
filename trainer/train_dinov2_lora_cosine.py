#!/usr/bin/env python3
import os, json, math, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from torch.utils.data import DataLoader

import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# local imports you already have
from config.opts import parse_args, add_lora_args
from data.dataset import NucleusCSV
from utils.utils import (
    set_seed, compute_metrics,
    init_metrics_logger, log_metrics, compute_per_class_metrics,
    print_per_class_summary, class_counts_from_df
)
from model.lora import apply_lora_to_timm_vit


# -----------------------------
# Cosine head with learnable temperature
# -----------------------------
class CosineHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, s_init: float = 20.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")
        self.s = nn.Parameter(torch.tensor(s_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        return self.s * (x @ w.t())


def monitor_temperature_and_adjustments(model: nn.Module, logit_adj: torch.Tensor, classes: List[str], epoch: int):
    """Print cosine temperature and the most extreme logit adjustments."""
    try:
        temp = float(model.head.s.item())
    except Exception:
        temp = float("nan")
    adj_values = logit_adj.squeeze().detach().cpu().numpy()
    min_idx = int(np.argmin(adj_values))
    max_idx = int(np.argmax(adj_values))
    print(f"  Cosine temperature: {temp:.2f} | Logit adj extremes: "
          f"{classes[max_idx]} (+{adj_values[max_idx]:.2f}), {classes[min_idx]} ({adj_values[min_idx]:.2f})")


class MacroF1EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 0.002):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -1.0
        self.counter = 0

    def __call__(self, macro_f1: float) -> bool:
        if macro_f1 > self.best_score + self.min_delta:
            self.best_score = macro_f1
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            print(f"\nEarly stopping: no macro F1 improvement for {self.patience} epochs")
            return True
        return False


def detailed_prediction_analysis(all_preds: torch.Tensor, all_targets: torch.Tensor,
                                 classes: List[str], epoch: int):
    """Every 10 epochs, show prediction vs truth share per class."""
    if epoch % 10 != 0 or all_preds.numel() == 0:
        return
    pred_counts = np.bincount(all_preds.cpu().numpy(), minlength=len(classes))
    true_counts = np.bincount(all_targets.cpu().numpy(), minlength=len(classes))
    total_pred, total_true = pred_counts.sum(), true_counts.sum()
    print("\nPrediction breakdown (every 10 epochs):")
    print(f"{'Class':<25} {'True %':<8} {'Pred %':<8} {'Ratio':<8} Status")
    for i, cls in enumerate(classes):
        tp = (true_counts[i] / total_true * 100.0) if total_true > 0 else 0.0
        pp = (pred_counts[i] / total_pred * 100.0) if total_pred > 0 else 0.0
        ratio = (pp / tp) if tp > 0 else 0.0
        status = "OK" if 0.7 <= ratio <= 1.4 else ("High" if ratio > 1.4 else "Low")
        print(f"{cls:<25} {tp:<8.1f} {pp:<8.1f} {ratio:<8.2f} {status}")




def is_main_process() -> bool:
    # single GPU script, always true
    return True


def main():
    args = parse_args(add_lora_args)
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load class map first ----
    with open(args.class_map, "r") as f:
        class_to_idx = json.load(f)
    classes = sorted(class_to_idx, key=lambda k: class_to_idx[k])
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")

    # ---- Build model (timm DINOv2 ViT by default) ----
    model = timm.create_model(
        args.model,
        pretrained=True,
        num_classes=num_classes,  # will be replaced by cosine head below
        img_size=args.img_size
    )

    # Get normalization from model cfg (fallback to ImageNet defaults)
    mean = tuple(model.pretrained_cfg.get('mean', IMAGENET_DEFAULT_MEAN))
    std  = tuple(model.pretrained_cfg.get('std',  IMAGENET_DEFAULT_STD))

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Replace linear head with cosine head, then unfreeze head
    in_dim = getattr(model.head, "in_features", getattr(model, "num_features"))
    model.head = CosineHead(in_dim, num_classes, s_init=20.0)
    for p in model.head.parameters():
        p.requires_grad = True

    # Inject LoRA adapters into last N blocks
    lora_params = apply_lora_to_timm_vit(
        model,
        last_n_blocks=args.lora_blocks,
        r=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,   # set this to 0.05 in your args for a touch of regularization
    )

    # Unfreeze norms in the last args.lora_blocks
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

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")

    # Print real train counts aligned to head order
    counts_np = class_counts_from_df(train_ds.df, label_col="cell_type", classes=classes)
    total = max(1, counts_np.sum())
    print("Train class distribution:")
    for cls, c in zip(classes, counts_np):
        pct = 100.0 * float(c) / float(total)
        print(f"  {cls:>24}: {int(c):7d} ({pct:5.1f}%)")

    # ---- DataLoaders (single GPU) ----
    pw = args.num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=pw
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=pw
    )

    # ---- Logit adjustment from priors (with tunable tau) ----
    prior = torch.from_numpy(counts_np.astype(np.float64) / counts_np.sum()).to(device=device, dtype=torch.float32)
    tau = float(getattr(args, "logit_tau", 1.0))  # if not in args, default to 1.0
    logit_adj = tau * torch.log(prior + 1e-12)    # shape [C]
    logit_adj = logit_adj.view(1, -1)             # shape [1, C]
    label_smoothing = 0.05
    print(f"\nLogit adjustment enabled with tau={tau:.2f}")
 
    # ---- Early stopping on macro F1 ----
    early_stopping = MacroF1EarlyStopping(patience=8, min_delta=0.002)

    # ---- Optimizer and Scheduler ----
    # Separate head weight vs temperature s so we can set weight decay on weight only
    head_weight_params, head_scale_params = [], []
    for n, p in model.head.named_parameters():
        if not p.requires_grad:
            continue
        if n == "s":
            head_scale_params.append(p)
        else:
            head_weight_params.append(p)

    optim_groups = [
        {"params": lora_params,            "lr": args.lr_lora, "weight_decay": args.weight_decay},
        {"params": head_weight_params,     "lr": args.lr_head, "weight_decay": args.weight_decay},
        {"params": head_scale_params,      "lr": args.lr_head, "weight_decay": 0.0},  # no decay on temperature
        {"params": norms,                  "lr": args.lr_head, "weight_decay": args.weight_decay},
    ]
    optimizer = torch.optim.AdamW(optim_groups)

    if args.cosine:
        total_epochs = args.epochs
        warmup = max(0, min(args.warmup_epochs, total_epochs - 1))

        def lr_lambda(epoch):
            if epoch < warmup:
                return float(epoch + 1) / float(max(1, warmup))
            progress = (epoch - warmup) / float(max(1, total_epochs - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16))

    # Save the used class map
    with open(Path(args.out_dir, "class_to_idx.used.json"), "w") as f:
        json.dump(class_to_idx, f, indent=2)

    best_macro_f1 = -1.0
    best_path = str(Path(args.out_dir, "best_dinov2_lora_cosine_logitadj.pt"))

    # Metrics logger
    run_id = time.strftime("%Y%m%d-%H%M%S")
    metrics_f, metrics_w = init_metrics_logger(args.out_dir, filename=f"metrics_{run_id}.csv", classes=classes)

    print("\nStarting training (single GPU)...")
    print("Using cosine head and logit adjusted cross entropy")
    print(f"LRs: head={args.lr_head}  lora={args.lr_lora}  | cosine_schedule={args.cosine} warmup={args.warmup_epochs}\n")

    params_to_clip = [p for g in optimizer.param_groups for p in g['params'] if p.requires_grad]

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=torch.cuda.is_available()):
                logits = model(x)                # cosine head returns logits
                loss = F.cross_entropy(logits + logit_adj, y, label_smoothing=label_smoothing)

            scaler.scale(loss).backward()

            # gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * y.size(0)
            c, t = compute_metrics(logits.detach(), y)
            train_correct += c
            train_total += t

        if scheduler is not None:
            scheduler.step()

        # ---- Validation ----
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0

        all_val_preds = []
        all_val_targets = []

        with torch.inference_mode(), torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=torch.cuda.is_available()):
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                loss = F.cross_entropy(logits + logit_adj, y, label_smoothing=label_smoothing)
                val_loss_sum += loss.item() * y.size(0)

                preds = logits.argmax(1)
                c, t = compute_metrics(logits, y)
                val_correct += c
                val_total += t

                all_val_preds.append(preds)
                all_val_targets.append(y)

        # Concatenate all predictions and targets
        if all_val_preds:
            all_preds = torch.cat(all_val_preds, dim=0).cpu()
            all_targets = torch.cat(all_val_targets, dim=0).cpu()
        else:
            all_preds = torch.empty(0, dtype=torch.long)
            all_targets = torch.empty(0, dtype=torch.long)

        # Aggregate metrics
        train_loss = train_loss_sum / max(1, train_total)
        train_acc  = train_correct / max(1, train_total)
        val_loss   = val_loss_sum   / max(1, val_total)
        val_acc    = val_correct    / max(1, val_total)
        secs = time.time() - t0

        per_class_metrics = None
        if len(all_preds) > 0:
            per_class_metrics, _, _ = compute_per_class_metrics(all_preds, all_targets, classes)

        # Compute macro F1 from per class metrics
        macro_f1 = None
        if per_class_metrics:
            macro_f1 = float(np.mean([m["f1"] for m in per_class_metrics]))
            print(f"Epoch {epoch:02d}/{args.epochs}  |  "
                  f"train_loss {train_loss:.4f}  train_acc {train_acc:.3f}  "
                  f"val_loss {val_loss:.4f}  val_acc {val_acc:.3f}  "
                  f"macro_f1 {macro_f1:.3f}  |  {secs:.1f}s")
            print_per_class_summary(per_class_metrics, classes)
         # Temperature and adjustment monitor (first 5 epochs, then every 5)
            if epoch <= 5 or epoch % 5 == 0:
                monitor_temperature_and_adjustments(model, logit_adj, classes, epoch)
            # Optional 10-epoch prediction breakdown
            detailed_prediction_analysis(all_preds, all_targets, classes, epoch)
        else:
            print(f"Epoch {epoch:02d}/{args.epochs}  |  "
                  f"train_loss {train_loss:.4f}  train_acc {train_acc:.3f}  "
                  f"val_loss {val_loss:.4f}  val_acc {val_acc:.3f}  |  {secs:.1f}s")

        # Log metrics CSV
        lr_lora = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else lr_lora
        log_metrics(
            metrics_w, run_id, epoch, train_loss, train_acc, val_loss, val_acc,
            lr_head, lr_lora, secs, per_class_metrics, classes
        )
        metrics_f.flush()

        # Save best by macro F1
        if macro_f1 is not None and macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            to_save = model.state_dict()
            torch.save({
                "model": to_save,
                "args": vars(args),
                "classes": classes,
                "val_acc": val_acc,
                "macro_f1": best_macro_f1,
            }, best_path)
            print(f"  ✓ Saved new best (macro F1={best_macro_f1:.3f}) to {best_path}")
            
        # Early stopping
        if macro_f1 is not None and early_stopping(macro_f1):
            break   
            

    if metrics_f is not None:
        metrics_f.close()
    print(f"\nDone. Best macro F1={best_macro_f1:.3f}")
    print(f"Checkpoint: {best_path}")


if __name__ == "__main__":
    main()
