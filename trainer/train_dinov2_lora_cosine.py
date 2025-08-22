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

from utils.utils import make_weighted_sampler
from sklearn.metrics import f1_score, classification_report

# local imports you already have
from config.opts import parse_args, add_lora_args
from data.dataset import NucleusCSV
from utils.utils import (
    set_seed, compute_metrics,
    init_metrics_logger, log_metrics, compute_per_class_metrics,
    print_per_class_summary, class_counts_from_df, maybe_init_wandb
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
        self._s_unconstrained = nn.Parameter(torch.tensor(s_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        s = F.softplus(self._s_unconstrained) + 1e-6# ensure s is positive
        return s * (x @ w.t())


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
    
    #print args
    print("Training DINOv2 with LoRA and Cosine Head")
    print("==> Launch config")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
     
            
    def _make_worker_init_fn(seed):
        def _init(worker_id):
            np.random.seed(seed + worker_id)
            torch.manual_seed(seed + worker_id)
        return _init

    worker_init = _make_worker_init_fn(args.seed)
    
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
    model.head = CosineHead(in_dim, num_classes, s_init=30.0)
    
    assert num_classes == model.head.weight.size(0), "Head out_dim does not match class map"
    
    # Freeze the temperature so it cannot drift
    with torch.no_grad():
        # keep s_init as the fixed scale
        model.head._s_unconstrained.copy_(torch.log(torch.expm1(torch.tensor(30.0))))
    model.head._s_unconstrained.requires_grad = False
    
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
    # Optional: WeightedRandomSampler for imbalance
    use_ws = bool(getattr(args, "use_weighted_sampler", False))
    label_col = "cell_type"
    if use_ws:
        sampler = make_weighted_sampler(train_ds.df, label_col=label_col)
        print(f"Using WeightedRandomSampler on '{label_col}'")
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,          # sampler implies shuffle must be False
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=worker_init,
            pin_memory=True,
            persistent_workers=pw,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=worker_init, 
            pin_memory=True, 
            persistent_workers=pw
        )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=worker_init,
        pin_memory=True, 
        persistent_workers=pw
    )

    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16))

    # ---- Logit adjustment from priors (with tunable tau) ----
    prior = torch.from_numpy((counts_np.astype(np.float64) / counts_np.sum()))
    prior = prior.clamp_min(1e-12).to(device=device, dtype=torch.float32)
    log_prior = torch.log(prior)
    tau = float(getattr(args, "logit_tau", 0.3))  # gentler than 1.0 or 0.5
    adj = (tau * log_prior).view(1, -1)
    adj = adj - adj.mean()  # centering keeps numerics tame without changing probabilities
    label_smoothing = 0.0   # see C
    print(f"\nLogit adjustment enabled with tau={tau:.2f}")
 
    # ---- Early stopping on macro F1 ----
    early_stopping = MacroF1EarlyStopping(patience=8, min_delta=0.002)

    # ---- Optimizer and Scheduler ----
    # Separate head weight vs temperature s so we can set weight decay on weight only
    head_weight_params, head_scale_params = [], []
    for n, p in model.head.named_parameters():
        if not p.requires_grad:
            continue
        if n == "_s_unconstrained":
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

    # Save the used class map
    with open(Path(args.out_dir, "class_to_idx.used.json"), "w") as f:
        json.dump(class_to_idx, f, indent=2)

    best_macro_f1 = -1.0
    best_path = str(Path(args.out_dir, "best_dinov2_lora_cosine_logitadj.pt"))

    # Metrics logger
    run_id = time.strftime("%Y%m%d-%H%M%S")
    metrics_f, metrics_w = init_metrics_logger(args.out_dir, filename=f"metrics_{run_id}.csv", classes=classes)
    
    wandb_mod, wb_run = maybe_init_wandb(args, classes, run_id, args.out_dir)

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
                logits_adj = logits - adj.to(dtype=logits.dtype)   # subtract log prior to favor tails
                loss = F.cross_entropy(logits_adj, y, label_smoothing=label_smoothing)

            scaler.scale(loss).backward()

            # gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * y.size(0)
            c, t = compute_metrics(logits_adj.detach(), y)
            train_correct += c
            train_total += t

        if scheduler is not None:
            scheduler.step()

        # ---- Validation ----
        model.eval()
        val_total = 0
        val_loss_sum = 0.0
        
        val_correct_raw = 0
        val_correct_adj = 0

        all_val_preds = []
        all_val_targets = []

        with torch.inference_mode(), torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=torch.cuda.is_available()):
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                logits_adj = logits - adj.to(dtype=logits.dtype)
                
                # loss for reporting: use RAW logits to keep validation consistent
                loss = F.cross_entropy(logits, y, label_smoothing=0)
                val_loss_sum += loss.item() * y.size(0)

                c_raw, t = compute_metrics(logits, y)
                c_adj, _ = compute_metrics(logits_adj, y)
                val_correct_raw += c_raw
                val_correct_adj += c_adj

                # predictions: use RAW for per-class metrics by default
                preds = logits.argmax(1)
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
        val_acc_raw = val_correct_raw / max(1, val_total)
        val_acc_adj = val_correct_adj / max(1, val_total)
        secs = time.time() - t0

        per_class_metrics = None
        if len(all_preds) > 0:
            per_class_metrics, _, _ = compute_per_class_metrics(all_preds, all_targets, classes)

         # Compute macro metrics from per class metrics (ignore NaNs)
        macro_f1 = None
        if per_class_metrics:
            macro_f1 = float(per_class_metrics.get("macro_f1", float("nan")))
            if np.isnan(macro_f1):
                macro_f1 = float(
                    f1_score(
                        all_targets.numpy(),
                        all_preds.numpy(),
                        average="macro",
                        zero_division=0,
                    )
                )
            print(f"Epoch {epoch:02d}/{args.epochs}  |  "
                  f"train_loss {train_loss:.4f}  train_acc {train_acc:.3f}  "
                  f"val_loss {val_loss:.4f}  val_acc_raw {val_acc_raw:.3f}  "
                  f"val_acc_adj {val_acc_adj:.3f}  "
                  f"macro_f1 {macro_f1 if macro_f1 is not None else float('nan'):.3f}  |  {secs:.1f}s")
            print_per_class_summary(per_class_metrics, classes)
         # Temperature and adjustment monitor (first 5 epochs, then every 5)
            if epoch <= 5 or epoch % 5 == 0:
                # show extremes of the actual subtraction term
                adj_vals = (-adj).squeeze().detach().cpu().numpy()  # this is what we add to logits effectively
                mi, ma = int(np.argmin(adj_vals)), int(np.argmax(adj_vals))
                temp = float(F.softplus(model.head._s_unconstrained).item())
                print(f"  Cosine temperature: {temp:.2f} | Adjustment extremes: "
                    f"{classes[ma]} ({adj_vals[ma]:+.2f}), {classes[mi]} ({adj_vals[mi]:+.2f})")
                
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
        
        # W&B logging
        if wb_run is not None:
            # macro_f1 may be None if per_class_metrics missing
            m_f1 = 0.0
            if 'macro_f1' in locals() and macro_f1 is not None and not np.isnan(macro_f1):
                m_f1 = float(macro_f1)
            wandb_mod.log({
                "epoch": epoch,
                "train/loss": train_loss, "train/acc": train_acc,
                "val/loss": val_loss, "val/acc": val_acc,
                "val/macro_f1": m_f1,
                "lr/head": lr_head, "lr/lora": lr_lora,
                "time/epoch_sec": secs
            }, step=epoch)


        # Save best by macro F1
        if macro_f1 is not None and not np.isnan(macro_f1) and macro_f1 > best_macro_f1:
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
        if macro_f1 is not None and not np.isnan(macro_f1) and early_stopping(macro_f1):
            break   
            

    if metrics_f is not None:
        metrics_f.close()
        
    if 'wb_run' in locals() and wb_run is not None:
        try:
            wb_run.finish()
        except Exception:
            pass
    
    print(f"\nDone. Best macro F1={best_macro_f1:.3f}")
    print(f"Checkpoint: {best_path}")


if __name__ == "__main__":
    main()
