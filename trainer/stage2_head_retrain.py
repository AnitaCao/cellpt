#!/usr/bin/env python3
import argparse, json, math, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader, WeightedRandomSampler

import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms

# project modules
from data.dataset import NucleusCSV
from model.lora import apply_lora_to_timm_vit

# ---- Cosine head (same as your trainer) ----
class CosineHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, s_init: float = 30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")
        self._s_unconstrained = nn.Parameter(torch.tensor(s_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        s = F.softplus(self._s_unconstrained) + 1e-6
        return s * (x @ w.t())

# ---- utils ----
@torch.no_grad()
def extract_prototypes(model, loader, num_classes, device):
    """Compute per-class normalized mean features using model.forward_features."""
    model.eval()
    # find feature dim
    feat_dim = None
    # quick dry-run to get feature dimension
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        feats = model.forward_features(xb)
        if feats.ndim > 2:
            feats = feats.mean(dim=(2,3))
        feat_dim = feats.shape[1]
        break

    protos = torch.zeros(num_classes, feat_dim, device=device)
    counts = torch.zeros(num_classes, device=device)
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        feats = model.forward_features(xb)
        if feats.ndim > 2:
            feats = feats.mean(dim=(2,3))
        feats = F.normalize(feats, dim=1)
        for c in yb.unique():
            m = (yb == c)
            if m.any():
                protos[c] += feats[m].sum(dim=0)
                counts[c] += m.sum()
    protos = protos / counts.clamp_min(1).unsqueeze(1)
    protos = F.normalize(protos, dim=1)
    return protos

def class_counts_from_df(df: pd.DataFrame, label_col: str, classes: list):
    vc = df[label_col].value_counts()
    return np.array([int(vc.get(c, 0)) for c in classes], dtype=np.int64)

def build_balanced_sampler(df: pd.DataFrame, label_col: str, classes: list, num_samples:int):
    """Balanced per-example weight = 1 / n_c (alpha=0)."""
    counts = class_counts_from_df(df, label_col, classes).astype(np.float64).clip(min=1.0)
    w_per_class = 1.0 / counts
    w_map = {c: w_per_class[i] for i, c in enumerate(classes)}
    w = torch.tensor([w_map[v] for v in df[label_col].tolist()], dtype=torch.float32)
    return WeightedRandomSampler(w, num_samples=num_samples, replacement=True)

def cosine_warmup_cosine(total_epochs, warmup):
    def lr_lambda(ep):
        if ep < warmup:
            return float(ep + 1) / float(max(1, warmup))
        prog = (ep - warmup) / float(max(1, total_epochs - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    return lr_lambda

def evaluate(model, loader, device):
    model.eval()
    all_y, all_p = [], []
    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            preds = logits.argmax(1).cpu().numpy()
            all_p.append(preds)
            all_y.append(yb.numpy())
    y = np.concatenate(all_y); p = np.concatenate(all_p)
    acc = accuracy_score(y, p)
    mf1 = f1_score(y, p, average="macro", zero_division=0)
    return acc, mf1

def main():
    ap = argparse.ArgumentParser("Stage-2 head-only retrain w/ prototype alignment + CB loss")
    ap.add_argument("--ckpt", required=True, help="Path to stage-1 checkpoint (.pt)")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--class_map", required=True)
    ap.add_argument("--label_col", default="cell_type")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--use_img_uniform", action="store_true")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--cb_beta", type=float, default=0.9999, help="Class-Balanced loss beta; set 0 to disable")
    ap.add_argument("--focal_gamma", type=float, default=0.0, help=">0 to use CB-Focal instead of CB-CE")
    ap.add_argument("--random_erasing_p", type=float, default=0.15)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load class map (ordering)
    with open(args.class_map, "r") as f:
        class_to_idx = json.load(f)
    classes = sorted(class_to_idx, key=lambda k: class_to_idx[k])
    num_classes = len(classes)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    ck_args = ckpt.get("args", {})
    model_name = ck_args.get("model", "vit_base_patch14_dinov2")
    img_size   = ck_args.get("img_size", 224)
    lora_blocks = ck_args.get("lora_blocks", 0)
    lora_rank   = ck_args.get("lora_rank", 16)
    lora_alpha  = ck_args.get("lora_alpha", 32.0)
    lora_dropout= ck_args.get("lora_dropout", 0.0)

    # Build model
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes, img_size=img_size)
    # freeze all
    for p in model.parameters(): p.requires_grad = False

    # replace head with cosine
    in_dim = getattr(model.head, "in_features", getattr(model, "num_features"))
    model.head = CosineHead(in_dim, num_classes, s_init=30.0)
    for p in model.head.parameters(): p.requires_grad = True

    # re-inject LoRA like Stage-1 (frozen)
    if lora_blocks and lora_blocks > 0:
        _ = apply_lora_to_timm_vit(model, last_n_blocks=lora_blocks, r=lora_rank,
                                   alpha=lora_alpha, dropout=lora_dropout)
        # keep LoRA frozen for stage-2
        for n, p in model.named_parameters():
            if "lora_" in n: p.requires_grad = False

    # load weights
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)

    # Build datasets (no augmentation for prototypes; light aug for head train)
    mean = tuple(model.pretrained_cfg.get('mean', IMAGENET_DEFAULT_MEAN))
    std  = tuple(model.pretrained_cfg.get('std',  IMAGENET_DEFAULT_STD))

    train_ds = NucleusCSV(args.train_csv, class_to_idx, mean, std, img_size,
                          label_col=args.label_col, use_img_uniform=args.use_img_uniform, augment=False)
    val_ds   = NucleusCSV(args.val_csv,   class_to_idx, mean, std, img_size,
                          label_col=args.label_col, use_img_uniform=args.use_img_uniform, augment=False)

    # Prototype alignment (no aug loader)
    proto_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    protos = extract_prototypes(model, proto_loader, num_classes, device)
    with torch.no_grad():
        model.head.weight.copy_(protos)

    # Stage-2 training loader: balanced sampler (alpha=0)
    num_samples = len(train_ds)
    sampler = build_balanced_sampler(train_ds.df, args.label_col, classes, num_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Loss: Class-Balanced CE (or CB-Focal)
    counts_np = class_counts_from_df(train_ds.df, args.label_col, classes).astype(np.float64).clip(min=1)
    if args.cb_beta > 0:
        eff_num = 1.0 - np.power(args.cb_beta, counts_np)
        class_w = ((1.0 - args.cb_beta) / eff_num)
        class_w = class_w / class_w.mean()
    else:
        class_w = np.ones_like(counts_np, dtype=np.float64)
    class_w_t = torch.tensor(class_w, dtype=torch.float32, device=device)

    def cb_loss(logits, y):
        if args.focal_gamma > 0:
            with torch.no_grad():
                pt = F.softmax(logits, dim=1).gather(1, y[:, None]).squeeze().clamp_min(1e-8)
            ce = F.cross_entropy(logits, y, weight=class_w_t, reduction="none", label_smoothing=0.0)
            loss = ((1 - pt) ** args.focal_gamma) * ce
            return loss.mean()
        else:
            return F.cross_entropy(logits, y, weight=class_w_t, label_smoothing=0.0)

    # Optimizer/scheduler: head-only
    head_weight_params, head_scale_params = [], []
    for name, p in model.head.named_parameters():
        if not p.requires_grad: continue
        if name == "_s_unconstrained": head_scale_params.append(p)
        else: head_weight_params.append(p)

    optim = torch.optim.AdamW([
        {"params": head_weight_params, "lr": args.lr_head, "weight_decay": args.weight_decay},
        {"params": head_scale_params,  "lr": args.lr_head, "weight_decay": 0.0},
    ])
    sched = torch.optim.lr_scheduler.LambdaLR(optim, cosine_warmup_cosine(args.epochs, args.warmup_epochs))

    # Light erasing? already off because augment=False in NucleusCSV. If you prefer, you can add a light head-only aug here.

    # Train head-only
    best_mf1 = -1.0
    best_path = Path(args.out_dir, "stage2_head.pt")
    print("\n[Stage-2] Head-only retrain starting…")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        # ensure only head requires grad
        for n, p in model.named_parameters():
            p.requires_grad_(n.startswith("head."))

        train_loss, nseen = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            logits = model(xb)  # cosine head applied
            loss = cb_loss(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 5.0)
            optim.step()

            bs = yb.size(0)
            train_loss += loss.item() * bs
            nseen += bs

        sched.step()
        train_loss /= max(1, nseen)

        acc, mf1 = evaluate(model, val_loader, device)
        secs = time.time() - t0
        print(f"Ep {ep:02d}/{args.epochs} | train_loss {train_loss:.4f} | val_acc {acc:.3f} | macro_f1 {mf1:.3f} | {secs:.1f}s")

        if mf1 > best_mf1:
            best_mf1 = mf1
            torch.save({
                "model": model.state_dict(),
                "args_stage2": vars(args),
                "classes": classes,
                "val_acc": acc,
                "macro_f1": best_mf1,
            }, best_path)
            print(f"  ✓ Saved new best head (macro_f1={best_mf1:.3f}) → {best_path}")

    print(f"\n[Stage-2] Done. Best macro F1={best_mf1:.3f} | Checkpoint: {best_path}")

if __name__ == "__main__":
    main()
