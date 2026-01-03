#!/usr/bin/env python3
import os, json, math, time, warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

import timm
from timm.data import Mixup
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV2

# project imports (unchanged)
from config.opts import parse_args, add_lora_args
from data.dataset import NucleusCSV
from utils.utils import (
    set_seed, compute_metrics,
    init_metrics_logger, log_metrics, compute_per_class_metrics,
    print_per_class_summary, class_counts_from_df, maybe_init_wandb, print_confusion_slice, save_confusion_matrices
)
from utils.losses import CBFocalLoss, LDAMLoss

from model.lora import apply_lora_to_timm_vit



# -----------------------------
# Cosine head with learnable temperature
# -----------------------------
class CosineHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, s_init: float = 30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")
        self._s_unconstrained = nn.Parameter(torch.tensor(s_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        s = F.softplus(self._s_unconstrained) + 1e-6  # ensure s > 0
        return s * (x @ w.t())


def evaluate_with_mask(model,
                       val_loader,
                       device,
                       amp_dtype,
                       adj,                    # epoch-wise logit adjustment tensor [1,C]
                       use_slide_mask: bool,
                       allowed_bitmask: dict,
                       classes: List[str]):
    """Run validation with 'effective' path (LA then optional slide mask) + 'raw' diagnostics.
       Returns dict with val_eff/acc, val_eff/loss, val_raw/acc, val_raw/loss, macro_f1, preds, targets."""
    model.eval()
    val_total = 0
    acc_eff = 0
    loss_eff_sum = 0.0
    acc_raw = 0
    loss_raw_sum = 0.0
    preds_all, targs_all = [], []

    with torch.inference_mode(), torch.amp.autocast('cuda', dtype=amp_dtype, enabled=torch.cuda.is_available()):
        for batch in val_loader:
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                x, y, sid_batch = batch
            else:
                x, y = batch
                sid_batch = None

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            x = x.to(dtype=next(model.parameters()).dtype)
            logits = model(x)

            # effective path: LA → mask
            logits_eff = logits - adj.to(dtype=logits.dtype) if adj is not None else logits
            if use_slide_mask:
                logits_eff = apply_slide_mask(logits_eff, sid_batch, allowed_bitmask)

            # effective metrics
            loss_eff = F.cross_entropy(logits_eff, y)
            loss_eff_sum += loss_eff.item() * y.size(0)
            c_eff, n_ = compute_metrics(logits_eff, y)
            acc_eff += c_eff
            val_total += n_
            preds_all.append(logits_eff.argmax(1).cpu())
            targs_all.append(y.cpu())

            # raw diagnostics
            loss_raw = F.cross_entropy(logits, y)
            loss_raw_sum += loss_raw.item() * y.size(0)
            c_raw, _ = compute_metrics(logits, y)
            acc_raw += c_raw

    P = torch.cat(preds_all) if preds_all else torch.empty(0, dtype=torch.long)
    T = torch.cat(targs_all) if targs_all else torch.empty(0, dtype=torch.long)
    macro_f1 = float(f1_score(T.numpy(), P.numpy(), average="macro", zero_division=0)) if len(P) > 0 else 0.0

    return {
        "val_eff/acc":  acc_eff / max(1, val_total),
        "val_eff/loss": loss_eff_sum / max(1, val_total),
        "val_raw/acc":  acc_raw / max(1, val_total),
        "val_raw/loss": loss_raw_sum / max(1, val_total),
        "macro_f1":     macro_f1,
        "preds":        P.numpy(),
        "targets":      T.numpy(),
    }

def log_to_wandb(wandb_mod, wb_run, epoch, base_metrics: dict,
                 ema_metrics: dict | None,
                 lrs: dict,
                 extra: dict | None = None):
    """Log standard metrics to W&B."""
    if wb_run is None or wandb_mod is None:
        return
    log = {
        "epoch": epoch,
        "train/loss": base_metrics["train_loss"],
        "train/acc": base_metrics["train_acc"],
        "val/loss_eff": base_metrics["val_loss_eff"],
        "val/acc_eff": base_metrics["val_acc_eff"],
        "val/loss_raw": base_metrics["val_loss_raw"],
        "val/acc_raw": base_metrics["val_acc_raw"],
        "val/macro_f1": base_metrics["macro_f1"],
        "la/tau_eff": base_metrics["tau_eff"],
        "la/label_smoothing": base_metrics["label_smoothing"],
        "lr/head": lrs.get("head", 0.0),
        "lr/head_w": lrs.get("head_w", 0.0),
        "lr/head_s": lrs.get("head_s", 0.0),
        "lr/lora": lrs.get("lora", 0.0),
        "lr/backbone": lrs.get("backbone", 0.0),
        "time/epoch_sec": base_metrics["epoch_secs"],
    }
    if ema_metrics is not None:
        log.update({
            "val_ema/acc_eff":  ema_metrics["val_eff/acc"],
            "val_ema/loss_eff": ema_metrics["val_eff/loss"],
            "val_ema/acc_raw":  ema_metrics["val_raw/acc"],
            "val_ema/loss_raw": ema_metrics["val_raw/loss"],
            "val_ema/macro_f1": ema_metrics["macro_f1"],
        })
    if extra:
        log.update(extra)
    try:
        wandb_mod.log(log, step=epoch)
    except Exception:
        pass


def replace_classifier_with_cosine(model: nn.Module, num_classes: int, s_init: float = 30.0) -> CosineHead:
    """
    Replace the backbone's classifier layer with a CosineHead in a model-agnostic way.
    Returns the CosineHead module (so we can build optim groups reliably).
    """
    # 1) infer input dim from current classifier if possible
    in_dim = None
    try:
        clf = model.get_classifier()
        if hasattr(clf, "in_features"):
            in_dim = clf.in_features
    except Exception:
        pass
    if in_dim is None:
        in_dim = getattr(model, "num_features", None)
    if in_dim is None:
        raise RuntimeError("Could not determine classifier in_features for this backbone.")

    cos_head = CosineHead(in_dim, num_classes, s_init=s_init)

    # 2) find classifier path from default_cfg if present
    name = None
    try:
        name = model.default_cfg.get("classifier", None)
    except Exception:
        name = None

    def _set_by_path(root, path: str, mod):
        obj = root
        parts = path.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], mod)

    replaced = False
    if name:
        try:
            _set_by_path(model, name, cos_head)
            replaced = True
        except Exception:
            replaced = False

    # 3) fallbacks
    if not replaced:
        if hasattr(model, "fc"):
            model.fc = cos_head
            replaced = True
        elif hasattr(model, "classifier"):
            model.classifier = cos_head
            replaced = True
        elif hasattr(model, "head") and isinstance(getattr(model, "head"), nn.Linear):
            model.head = cos_head
            replaced = True

    if not replaced:
        # Last resort: try reset_classifier if available, then set by attr
        try:
            model.reset_classifier(num_classes=num_classes)
        except Exception:
            pass
        # try again quickly
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            model.fc = cos_head
            replaced = True
        elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
            model.classifier = cos_head
            replaced = True
        elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
            model.head = cos_head
            replaced = True

    if not replaced:
        raise RuntimeError("Failed to replace classifier with CosineHead for this backbone.")

    # Save a stable alias; DO NOT overwrite the model's structural 'head' module (e.g., ConvNeXt head)
    setattr(model, "_cosine_head", cos_head)
    return cos_head


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


def apply_slide_mask(logits, slide_ids, mask_table):
    if slide_ids is None:
        return logits
    out = logits.clone()
    neg_inf = torch.finfo(out.dtype).min
    for i, sid in enumerate(slide_ids):
        m = mask_table.get(sid, None)
        if m is None:
            continue  # unknown slide_id -> do not mask
        m = m.to(device=out.device)
        out[i, ~m] = neg_inf
    return out


def is_main_process() -> bool:
    return True


def main():
    args = parse_args(add_lora_args)

    print("Training DINOv2 (or other timm backbone) with LoRA (ViT only) and Cosine Head")
    print("==> Launch config")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # ---- Env toggles ----
 
    # EMA
    ema_on      = os.environ.get("EMA", "0") == "1"
    ema_decay   = float(os.environ.get("EMA_DECAY", "0.9999"))
    ema_eval    = os.environ.get("EMA_EVAL", "1") == "1"
    # Data shaping
    smooth_on       = os.environ.get("SMOOTH_SAMPLER", "0") == "1"
    smooth_alpha    = float(os.environ.get("SMOOTH_ALPHA", "0.5"))   # 0.5 ~ sqrt
    epoch_budget    = int(os.environ.get("EPOCH_BUDGET", "0"))
    downsample_frac = float(os.environ.get("DOWNSAMPLE_HEAD_FRAC", "1.0"))  # e.g., 0.8
    balanced_last_k = int(os.environ.get("BALANCED_LAST_K", "0"))           # e.g., 3

    def _make_worker_init_fn(seed):
        def _init(worker_id):
            np.random.seed(seed + worker_id)
            torch.manual_seed(seed + worker_id)
        return _init

    def _find_group_lr(optimizer, param_set):
        for g in optimizer.param_groups:
            if any(p in param_set for p in g["params"]):
                return g["lr"]
        return None

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

    # sanity: contiguous 0..K-1 indices
    idxs = sorted(class_to_idx.values())
    if idxs != list(range(len(idxs))):
        warnings.warn(f"[label-map] Indices are not contiguous 0..{len(idxs)-1}. Got: {idxs[:min(10,len(idxs))]} ...")

    # ---- Build model ----
    model = timm.create_model(
        args.model,
        pretrained=True,
        num_classes=num_classes,  # will be replaced by cosine head immediately
        img_size=args.img_size
    )

    # Get normalization from model cfg (fallback to ImageNet defaults)
    try:
        mean = tuple(model.pretrained_cfg.get('mean', IMAGENET_DEFAULT_MEAN))
        std  = tuple(model.pretrained_cfg.get('std',  IMAGENET_DEFAULT_STD))
    except Exception:
        mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    # Freeze or unfreeze
    unfreeze = bool(getattr(args, "unfreeze_backbone", False))
    if unfreeze:
        for p in model.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = False

    # Replace classifier with cosine head (backbone-agnostic), then unfreeze head params
    cos_head = replace_classifier_with_cosine(model, num_classes, s_init=30.0)
    assert num_classes == cos_head.weight.size(0), "Head out_dim does not match class map"
    for p in cos_head.parameters():
        p.requires_grad = True


    with torch.no_grad():
        target_s = torch.tensor(30.0)
        cos_head._s_unconstrained.copy_(torch.log(torch.expm1(target_s)))
    cos_head._s_unconstrained.requires_grad = True

    # LoRA (ViT only; skipped for ConvNeXt / Swin)
    is_vit = hasattr(model, "blocks") and len(getattr(model, "blocks", [])) > 0
    lora_params: List[torch.nn.Parameter] = []
    if not unfreeze and is_vit and getattr(args, "lora_blocks", 0) > 0:
        lora_params = apply_lora_to_timm_vit(
            model,
            last_n_blocks=args.lora_blocks,
            r=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )

    # Unfreeze norms in the last args.lora_blocks (ViT only)
    norms = []
    if not unfreeze and is_vit and getattr(args, "lora_blocks", 0) > 0:
        total_blocks = len(model.blocks)
        start = max(0, total_blocks - args.lora_blocks)
        for i in range(start, total_blocks):
            for name in ("norm1", "norm2"):
                m = getattr(model.blocks[i], name, None)
                if m is not None:
                    for p in m.parameters():
                        p.requires_grad = True
                    norms += list(m.parameters())

    model.to(device)

    try:
        n_lora = sum(p.numel() for p in lora_params)
        if n_lora > 0:
            print(f"LoRA trainable params: {n_lora}")
    except Exception:
        pass

    # -------- Mixup / CutMix wiring --------
    mixup_alpha   = float(getattr(args, "mixup_alpha", 0.0))
    cutmix_alpha  = float(getattr(args, "cutmix_alpha", 0.0))
    mixup_prob    = float(getattr(args, "mixup_prob", 1.0))
    mixup_switch  = float(getattr(args, "mixup_switch_prob", 0.0))
    mixup_mode    = str(getattr(args, "mixup_mode", "batch"))
    mixup_off_k   = int(getattr(args, "mixup_off_epochs", 10))
    mixup_enabled = (mixup_alpha > 0.0 or cutmix_alpha > 0.0) 
    mixup_fn = None
    soft_ce = None
    
    if mixup_enabled:
        mixup_fn = Mixup(
            mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha,
            prob=mixup_prob, switch_prob=mixup_switch, mode=mixup_mode,
            label_smoothing=0.0, num_classes=num_classes
        )
        soft_ce = SoftTargetCrossEntropy()
        print(f"Mixup/CutMix enabled: mixup_alpha={mixup_alpha} cutmix_alpha={cutmix_alpha} ")

    label_col = getattr(args, "label_col", "cell_type")
    print(f"Using label column: {label_col}")
    print(f"Base sampler policy: {args.sampler.upper()}")

    # ---- Datasets ----
    train_ds = NucleusCSV(
        args.train_csv, class_to_idx, size=args.img_size, mean=mean, std=std,
        label_col=label_col, use_img_uniform=args.use_img_uniform, augment=True,
        return_slide=True
    )
    val_ds = NucleusCSV(
        args.val_csv, class_to_idx, size=args.img_size, mean=mean, std=std,
        label_col=label_col, use_img_uniform=args.use_img_uniform, augment=False,
        return_slide=True
    )
    
    
    for ds_name, ds in (("train", train_ds), ("val", val_ds)):
        before = len(ds.df)
        ds.df = ds.df[ds.df[label_col].isin(class_to_idx.keys())].reset_index(drop=True)
        removed = before - len(ds.df)
        if removed > 0:
            print(f"[filter] {ds_name}: removed {removed} rows whose labels are not in class_map")

    print(f"Train samples: {len(train_ds)}")
    if unfreeze:
        print(f"Full finetune mode: unfreeze_backbone=1  lr_backbone={args.lr_backbone}")
    print(f"Val samples:   {len(val_ds)}")
    
    
     # --- Optional per-class cap to control head/tail ratio (TRAIN only) ---
    cap_ratio = float(getattr(args, "cap_ratio", 0.0))
    cap_base  = str(getattr(args, "cap_base", "min"))
    if cap_ratio > 0.0:
        vc0 = train_ds.df[label_col].value_counts()
        counts = np.array([int(vc0.get(c, 0)) for c in classes], dtype=np.int64)
        nonzero = counts[counts > 0]
        if nonzero.size == 0:
            raise RuntimeError("All train class counts are zero after JSON filtering?")
        if cap_base == "min":
            base = int(nonzero.min())
        elif cap_base == "p1":
            base = int(max(1, math.floor(np.percentile(nonzero, 1))))
        elif cap_base == "p5":
            base = int(max(1, math.floor(np.percentile(nonzero, 5))))
        elif cap_base == "p10":
            base = int(max(1, math.floor(np.percentile(nonzero, 10))))
        else:
            base = int(nonzero.min())
        cap = int(max(1, math.floor(cap_ratio * base)))

        before = {c: int(vc0.get(c, 0)) for c in classes}
        capped_parts = []
        for c in classes:
            cdf = train_ds.df[train_ds.df[label_col] == c]
            if len(cdf) > cap:
                cdf = cdf.sample(cap, random_state=args.seed)
            capped_parts.append(cdf)
        train_ds.df = (
            pd.concat(capped_parts, axis=0)
              .sample(frac=1.0, random_state=args.seed)
              .reset_index(drop=True)
        )
        vc1 = train_ds.df[label_col].value_counts()
        after = {c: int(vc1.get(c, 0)) for c in classes}
        print(f"[cap] Per-class cap active: base={cap_base}({base}) ratio={cap_ratio} → cap={cap}")
        print(f"[cap] Train size: {sum(before.values())} → {len(train_ds.df)}")
        for c in classes:
            if after[c] != before[c]:
                print(f"  [cap] {c}: {before[c]} → {after[c]}")

    # Optional downsample of the head (largest) class in training only
    if downsample_frac < 1.0:
        vc0 = train_ds.df[label_col].value_counts()
        head_cls = vc0.idxmax()
        n0 = int(vc0.max())
        keep = max(1, int(n0 * downsample_frac))
        head_keep = train_ds.df[train_ds.df[label_col] == head_cls].sample(keep, random_state=args.seed)
        rest = train_ds.df[train_ds.df[label_col] != head_cls]
        train_ds.df = pd.concat([head_keep, rest]).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        print(f"[downsample] Head '{head_cls}': {n0} -> {keep} (frac={downsample_frac}) | New train size: {len(train_ds.df)}")

    # Show class distribution (aligned to head order)
    counts_np = class_counts_from_df(train_ds.df, label_col=label_col, classes=classes)
    total = max(1, counts_np.sum())
    print("Train class distribution:")
    for cls, c in zip(classes, counts_np):
        pct = 100.0 * float(c) / float(total)
        print(f"  {cls:>24}: {int(c):7d} ({pct:5.1f}%)")

    # ---- DataLoaders ----
    pin = (device.type == "cuda")
    pw_flag = args.num_workers > 0 and pin

    
    # UNIFORM (shuffle) base loader
    train_loader_shuffle = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, worker_init_fn=worker_init,
        pin_memory=pin, persistent_workers=pw_flag, drop_last=True
    )
    
    # WEIGHTED (1/n_c) base loader
    train_loader_weighted = None
    if args.sampler == "weighted":
        vc = train_ds.df[label_col].value_counts()
        n = np.array([int(vc.get(c, 0)) for c in classes], dtype=np.float64)
        w_per_class = 1.0 / np.clip(n, 1.0, None)
        w_map = {c: w_per_class[i] for i, c in enumerate(classes)}
        w = torch.tensor([w_map[v] for v in train_ds.df[label_col].tolist()], dtype=torch.float32)
        sampler_w = WeightedRandomSampler(w, num_samples=len(train_ds.df), replacement=True)
        train_loader_weighted = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler_w, shuffle=False,
            num_workers=args.num_workers, worker_init_fn=worker_init,
            pin_memory=pin, persistent_workers=pw_flag, drop_last=True
        )
        print("[sampler] Using WEIGHTED sampler (1/n_c).")
    else:
        print("[sampler] Using UNIFORM sampler (shuffle).")
    
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, worker_init_fn=worker_init,
        pin_memory=pin, persistent_workers=pw_flag
    )
    
    #Build slide‑level allowed class bitmasks for constrained inference
    slide_to_allowed = (
        train_ds.df.groupby("slide_id")[label_col]
        .apply(lambda s: sorted(s.unique().tolist()))
        .to_dict()
    )
    
    allowed_bitmask = {}
    
    for sid, cls_list in slide_to_allowed.items():
        m = np.zeros(len(classes), dtype=bool)
        for c in cls_list:
            m[class_to_idx[c]] = True
        allowed_bitmask[sid] = torch.from_numpy(m)

    print(f"Built context masks for {len(allowed_bitmask)} slides")

    # Power-smoothing sampler
    if smooth_on :
        vc = train_ds.df[label_col].value_counts()
        n = np.array([int(vc.get(c, 0)) for c in classes], dtype=np.float64).clip(1.0, None)
        w_per_class = n ** (smooth_alpha - 1.0)
        w_map = {c: w_per_class[i] for i, c in enumerate(classes)}
        w = torch.tensor([w_map[cell_type] for cell_type in train_ds.df[label_col]], dtype=torch.float32)
        num_samples = epoch_budget if epoch_budget > 0 else len(train_ds)
        implied = (n * w_per_class); implied = implied / implied.sum()
        head_i = int(np.argmax(n)); tail_i = int(np.argmin(n))
        print(f"Smoothing alpha={smooth_alpha:.2f} implied share  "
              f"head {classes[head_i]}={implied[head_i]:.4f}  "
              f"tail {classes[tail_i]}={implied[tail_i]:.4f}  "
              f"budget={num_samples if epoch_budget>0 else 'full'}")
        smooth_sampler = WeightedRandomSampler(w, num_samples=num_samples, replacement=True)
        train_loader_smooth = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=smooth_sampler, shuffle=False,
            num_workers=args.num_workers, worker_init_fn=worker_init,
            pin_memory=pin, persistent_workers=pw_flag, drop_last=True
        )
    else:
        train_loader_smooth = None

    # Optional: balanced sampler for the last K epochs only
    balanced_loader = None
    if balanced_last_k > 0:
        vc = train_ds.df[label_col].value_counts()
        n = np.array([int(vc.get(c, 0)) for c in classes], dtype=np.float64)
        w_per_class = 1.0 / np.clip(n, 1.0, None)
        w_map = {c: w_per_class[i] for i, c in enumerate(classes)}
        w = torch.tensor([w_map[v] for v in train_ds.df[label_col].tolist()], dtype=torch.float32)
        sampler_bal = WeightedRandomSampler(w, num_samples=len(train_ds.df), replacement=True)
        balanced_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler_bal, shuffle=False,
            num_workers=args.num_workers, worker_init_fn=worker_init,
            pin_memory=pin, persistent_workers=pw_flag, drop_last=True
        )
        print(f"[sampler] Balanced sampler will be used for the last {balanced_last_k} epochs.")

    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16))

    # ---- Priors / LA ----
    prior_counts_np = counts_np
    prior_csv = getattr(args, "prior_csv", None)
    if prior_csv is not None and str(prior_csv).strip():
        try:
            cdf = pd.read_csv(prior_csv)
            cand_cols = ["img_path_uniform", "img_path"]
            col = next((c for c in cand_cols if c in cdf.columns), None)
            if col is None:
                raise RuntimeError("No image path column found in prior_csv.")
            cdf = cdf[(cdf[col].map(lambda p: Path(str(p)).is_file()))]
            cdf = cdf[cdf[label_col].isin(class_to_idx.keys())]
            vc = cdf[label_col].value_counts()
            prior_counts_np = np.array([int(vc.get(c, 0)) for c in classes], dtype=np.int64)
            print(f"Using priors from prior_csv={prior_csv}")
        except Exception as e:
            print(f"Warning: failed to read prior_csv ({e}); falling back to train counts.")
            prior_counts_np = counts_np

    prior = torch.from_numpy(prior_counts_np.astype(np.float64) / max(1, prior_counts_np.sum()))
    prior = prior.clamp_min(1e-12).to(device=device, dtype=torch.float32)
    log_prior = torch.log(prior)

    early_stop_off = os.environ.get("EARLY_STOP_OFF", "0") == "1"
    tau_target = float(getattr(args, "logit_tau", 1.0))
    if os.environ.get("DISABLE_LA", "0") == "1":
        tau_target = 0.0
    la_start_epoch = int(getattr(args, "la_start_epoch", 3))
    la_ramp_epochs = int(getattr(args, "la_ramp_epochs", 3))

    print(f"DISABLE_LA={'1' if tau_target==0.0 else '0'}  EARLY_STOP_OFF={'1' if early_stop_off else '0'}")
    if smooth_on:
        eb_text = epoch_budget if epoch_budget > 0 else "full"
        print(f"Power smoothing enabled: alpha={smooth_alpha:.3f}  epoch_budget={eb_text}")
    print(f"\nLogit adjustment configured: tau_target={tau_target:.2f}, "
          f"la_start_epoch={la_start_epoch}, la_ramp_epochs={la_ramp_epochs}")

    # Default label smoothing (used when mixup is OFF)
    ls_target = float(getattr(args, "label_smoothing", 0.0))


    # ---- Imbalance-aware loss selection ----
    # Avoid combining weighted base sampling with CB/LDAM
    if args.loss_type in ["cb_focal", "ldam_drw"] and args.sampler == "weighted":
        print("[warning] Avoid combining weighted sampler with CB-Focal/LDAM-DRW; switching base sampler to UNIFORM.")
        args.sampler = "uniform"

    cls_counts = counts_np.astype(np.int64)
    cb_loss = None
    ldam_loss = None
    if args.loss_type == "cb_focal":
        cb_loss = CBFocalLoss(cls_counts, beta=args.cb_beta, gamma=args.focal_gamma).to(device)
    elif args.loss_type == "ldam_drw":
        ldam_loss = LDAMLoss(cls_counts, max_m=args.ldam_max_m, s=args.ldam_s,
                             drw=True, beta=args.cb_beta, drw_start=args.drw_start_epoch).to(device)

    '''
    # Mixup only with CE
    if args.loss_type != "ce" and mixup_enabled:
        print("[loss] Disabling mixup/cutmix since loss_type != 'ce'.")
        mixup_enabled = False
        mixup_fn = None
        soft_ce = None
    '''
    # Disable LA for non‑CE losses
    if args.loss_type != "ce" and tau_target > 0.0:
        print("[LA] Disabled for non‑CE loss.")
        tau_target = 0.0

    # ---- Optimizer & Scheduler ----
    # Separate head weight vs temperature s so we can set weight decay on weight only
    head_weight_params, head_scale_params = [], []
    for n, p in getattr(model, "_cosine_head").named_parameters():
        if not p.requires_grad:
            continue
        if n == "_s_unconstrained":
            head_scale_params.append(p)
        else:
            head_weight_params.append(p)


    if unfreeze:
        # backbone = all trainable params except cosine head
        head_param_set = set(p for p in getattr(model, "_cosine_head").parameters())
        backbone_params = [p for p in model.parameters() if p.requires_grad and p not in head_param_set]
        optim_groups = [
            {"params": backbone_params,    "lr": args.lr_backbone, "weight_decay": 5e-2},
            {"params": head_weight_params, "lr": args.lr_head,     "weight_decay": 1e-4},
            {"params": head_scale_params,  "lr": args.lr_head,     "weight_decay": 0.0},
        ]
    else:
        wd = getattr(args, "weight_decay", 5e-2)
        optim_groups = [
            {"params": lora_params,        "lr": args.lr_lora, "weight_decay": 0.0},
            {"params": head_weight_params, "lr": args.lr_head, "weight_decay": 1e-4},
            {"params": head_scale_params,  "lr": args.lr_head, "weight_decay": 0.0},
            {"params": norms,              "lr": args.lr_head * 0.2, "weight_decay": wd},
        ]
    optimizer = torch.optim.AdamW(optim_groups)

    # EMA (track full model)
    ema_model = ModelEmaV2(model, decay=ema_decay, device=device) if ema_on else None
    if ema_model is not None:
        print(f"[EMA] Enabled (decay={ema_decay}, eval={'EMA' if ema_eval else 'student'})")

    # LR scheduler
    if args.cosine:
        total_epochs = args.epochs
        warmup = max(0, min(args.warmup_epochs, total_epochs - 1))

        def lr_lambda(epoch_idx):
            if epoch_idx < warmup:
                return float(epoch_idx + 1) / float(max(1, warmup))
            progress = (epoch_idx - warmup) / float(max(1, total_epochs - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    # Persist class map used
    with open(Path(args.out_dir, "class_to_idx.used.json"), "w") as f:
        json.dump(class_to_idx, f, indent=2)
    
    run_id = time.strftime("%Y%m%d-%H%M%S")

    # Metrics + W&B
    best_metric = -1.0
    #create path file name based on model and settings
    file_name = f"{args.model}_bs{args.batch_size}_lrh{args.lr_head}_{run_id}.pt"
    best_path = str(Path(args.out_dir, file_name))
    save_by = getattr(args, "save_metric", "acc")  # "acc" or "f1"
    
    metrics_f, metrics_w = init_metrics_logger(args.out_dir, filename=f"metrics_{run_id}.csv", classes=classes)
    wandb_mod, wb_run = maybe_init_wandb(args, classes, run_id, args.out_dir)

    print("\nStarting training (single GPU)...")
    print("Using cosine head and " + ("logit-adjusted CE (with ramp)" if tau_target>0 else "plain CE (no LA)"))
    print(f"LRs: head={args.lr_head}  lora={getattr(args,'lr_lora',0.0)}  | cosine_schedule={args.cosine} warmup={args.warmup_epochs}\n")

    params_to_clip = [p for g in optimizer.param_groups for p in g['params'] if p.requires_grad]

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        # Mixup schedule
        use_mixup_this_epoch = mixup_enabled and (epoch <= (args.epochs - max(0, mixup_off_k)))
        if mixup_enabled and epoch == 1:
            print(f"[epoch {epoch}] mixup/cutmix: ON (alpha={mixup_alpha}, cutmix={cutmix_alpha}, prob={mixup_prob}, mode={mixup_mode})")
        if mixup_enabled and epoch == (args.epochs - max(0, mixup_off_k) + 1):
            print(f"[epoch {epoch}] mixup/cutmix: OFF for final {mixup_off_k} epochs")


        use_balanced_now = (balanced_last_k > 0 and epoch > (args.epochs - balanced_last_k) and balanced_loader is not None)
        if use_balanced_now:
            train_loader = balanced_loader
        elif smooth_on and train_loader_smooth is not None:
            train_loader = train_loader_smooth
        else:
            train_loader = (
                train_loader_weighted
                if (args.sampler == "weighted" and train_loader_weighted is not None)
                else train_loader_shuffle
            )

        # ---- build epoch-wise LA & smoothing ----
        if epoch < la_start_epoch:
            tau_eff = 0.0
        else:
            prog = min(1.0, max(0.0, (epoch - la_start_epoch + 1) / max(1, la_ramp_epochs)))
            tau_eff = tau_target * prog

        adj = (tau_eff * log_prior).view(1, -1)
        adj = adj - adj.mean()

        # Label smoothing: when mixup is ON → soft loss; otherwise use configured LS
        label_smoothing = 0.0 if use_mixup_this_epoch else ls_target
        if epoch <= 3 or epoch % 5 == 0:
            print(f"[epoch {epoch}] tau_eff={tau_eff:.3f}, label_smoothing={label_smoothing:.3f}")

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            if len(batch) == 3:
                x, y, sid_batch = batch
            else:
                x, y = batch; sid_batch = None
            
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x = x.to(dtype=next(model.parameters()).dtype)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=torch.cuda.is_available()):
                y_hard = y.clone()
                if use_mixup_this_epoch and mixup_fn is not None:
                    x, y = mixup_fn(x, y)  # y becomes soft targets

                  
                logits = model(x)
                logits_eff = logits - adj.to(dtype=logits.dtype) if tau_eff > 0 else logits

                # ---- Loss switch ----
                if args.loss_type == "ce":
                    if use_mixup_this_epoch and soft_ce is not None:
                        loss = soft_ce(logits_eff, y)  # soft targets
                    else:
                        loss = F.cross_entropy(logits_eff, y, label_smoothing=label_smoothing)
                elif args.loss_type == "cb_focal":
                    loss = cb_loss(logits_eff, y_hard)
                elif args.loss_type == "ldam_drw":
                    loss = ldam_loss(logits_eff, y_hard, epoch=epoch)
                else:
                    raise ValueError(f"Unknown loss_type: {args.loss_type}")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            if ema_model is not None:
                ema_model.update(model)

            train_loss_sum += loss.item() * y_hard.size(0)
            c, t_ = compute_metrics(logits_eff.detach(), y_hard)
            train_correct += c
            train_total += t_

        if scheduler is not None:
            scheduler.step()

        # ---- Validation (student) ----
        model.eval()
        val_total = 0
        val_loss_raw_sum = 0.0
        val_loss_adj_sum = 0.0
        val_correct_raw = 0
        val_correct_adj = 0
        all_val_preds, all_val_targets = [], []

        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=amp_dtype, enabled=torch.cuda.is_available()):
            for batch in val_loader:
                if len(batch) == 3:
                    x, y, sid_batch = batch
                else:
                    x, y = batch; sid_batch = None
              
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                
                x = x.to(dtype=next(model.parameters()).dtype)
                
                logits = model(x)
                logits_eff = logits - adj.to(dtype=logits.dtype) if tau_eff > 0 else logits
                if args.use_slide_mask:
                    logits_eff = apply_slide_mask(logits_eff, sid_batch, allowed_bitmask)

                # Effective metrics (the ones you care about)
                loss_eff = F.cross_entropy(logits_eff, y)
                val_loss_adj_sum += loss_eff.item() * y.size(0)
                c_eff, t_ = compute_metrics(logits_eff, y)
                val_correct_adj += c_eff
                val_total += t_
                preds_eff = logits_eff.argmax(1)

                # Raw (optional diagnostics)
                loss_raw = F.cross_entropy(logits, y)
                val_loss_raw_sum += loss_raw.item() * y.size(0)
                c_raw, _ = compute_metrics(logits, y)
                val_correct_raw += c_raw
                
                all_val_preds.append(preds_eff)
                all_val_targets.append(y)

        if all_val_preds:
            all_preds = torch.cat(all_val_preds, dim=0).cpu()
            all_targets = torch.cat(all_val_targets, dim=0).cpu()
        else:
            all_preds = torch.empty(0, dtype=torch.long)
            all_targets = torch.empty(0, dtype=torch.long)

        train_loss = train_loss_sum / max(1, train_total)
        train_acc = train_correct / max(1, train_total)
        val_loss_raw = val_loss_raw_sum / max(1, val_total)
        val_loss_adj = val_loss_adj_sum / max(1, val_total)
        val_acc_raw = val_correct_raw / max(1, val_total)
        val_acc_adj = val_correct_adj / max(1, val_total)
        secs = time.time() - t0
        
        

        per_class_metrics = None
        if len(all_preds) > 0:
            per_class_metrics, _, _ = compute_per_class_metrics(all_preds, all_targets, classes)

        macro_f1 = None
        if per_class_metrics:
            macro_f1 = float(per_class_metrics.get("macro_f1", float("nan")))
            if np.isnan(macro_f1):
                macro_f1 = float(
                    f1_score(all_targets.numpy(), all_preds.numpy(), average="macro", zero_division=0)
                )

            print(f"Epoch {epoch:02d}/{args.epochs}  |  "
                  f"train_loss {train_loss:.4f}  train_acc {train_acc:.3f}  "
                  f"val_loss_adj {val_loss_adj:.4f}  val_acc_adj {val_acc_adj:.3f}  "
                  f"(val_loss_raw {val_loss_raw:.4f}  val_acc_raw {val_acc_raw:.3f})  "
                  f"macro_f1 {macro_f1 if macro_f1 is not None else float('nan'):.3f}  "
                  f"| tau_eff {tau_eff:.3f}  ls {label_smoothing:.3f} |  {secs:.1f}s")
            print_per_class_summary(per_class_metrics, classes)
            # Temperature & adj monitor
            if epoch <= 3 or epoch % 5 == 0:
                adj_vals = (-(adj)).squeeze().detach().cpu().numpy()
                mi, ma = int(np.argmin(adj_vals)), int(np.argmax(adj_vals))
                temp = float(F.softplus(getattr(model, "_cosine_head")._s_unconstrained).item())
                print(f"  Cosine temperature: {temp:.2f} | Adjustment extremes: "
                      f"{classes[ma]} ({adj_vals[ma]:+.2f}), {classes[mi]} ({adj_vals[mi]:+.2f})")
            # Optional every 10 epochs
            detailed_prediction_analysis(all_preds, all_targets, classes, epoch)
            if epoch % 10 == 0 or epoch == args.epochs:
                #hard = ["Myeloid cells", "T cells", "Pericytes"]  # hard for fine labels
                #hard = ["Stromal cells", "Epithelial cells", "Endothelial cells"] #hard for coarse labels
                #print_confusion_slice(all_targets.numpy(), all_preds.numpy(), classes, hard, topk=4)
                
                cm_paths = save_confusion_matrices(
                    all_targets.numpy(),
                    all_preds.numpy(),
                    classes,
                    args.out_dir,
                    epoch=epoch,
                    norms=("true", "none", "pred"),      # or ("none","true","pred","all")
                    print_table_for="true",      # prints one table to stdout
                    annotate=True
                )
                
                # Log confusion PNGs to W&B if available
                if wb_run is not None:
                    try:
                        for tag, paths in cm_paths.items():
                            wandb_mod.log(
                                {f"val/confusion_{tag}": wandb_mod.Image(str(paths["png"])), "epoch": epoch},
                                step=epoch
                            )
                    except Exception:
                        pass
                
        else:
            print(f"Epoch {epoch:02d}/{args.epochs}  |  "
                  f"train_loss {train_loss:.4f}  train_acc {train_acc:.3f}  "
                  f"val_loss_adj {val_loss_adj:.4f}  val_acc_adj {val_acc_adj:.3f}  "
                  f"(val_loss_raw {val_loss_raw:.4f}  val_acc_raw {val_acc_raw:.3f})  "
                  f"| tau_eff {tau_eff:.3f}  ls {label_smoothing:.3f} |  {secs:.1f}s")

        # ---- EMA eval (optional) ----
        ema_metrics = None
        if ema_model is not None and ema_eval:
            # backup student
            student_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            # EMA weights
            model.load_state_dict(ema_model.module.state_dict(), strict=False)
            ema_metrics = evaluate_with_mask(
                model, val_loader, device, amp_dtype,
                adj=(adj if tau_eff > 0 else None),
                use_slide_mask=bool(getattr(args, "use_slide_mask", False)),
                allowed_bitmask=allowed_bitmask,
                classes=classes
            )
            # restore student
            model.load_state_dict({k: v.to(device) for k, v in student_sd.items()}, strict=False)


        # ---- Logging ----
        lr_head = _find_group_lr(optimizer, set(head_weight_params)) or 0.0
        lr_lora = _find_group_lr(optimizer, set(lora_params) if isinstance(lora_params, list) else set()) or 0.0
        lr_backbone = 0.0
        if unfreeze:
            head_param_set = set(p for p in getattr(model, "_cosine_head").parameters())
            backbone_set = set(p for p in model.parameters() if p.requires_grad and p not in head_param_set)
            lr_backbone = _find_group_lr(optimizer, backbone_set) or 0.0

        log_metrics(
            metrics_w, run_id, epoch, train_loss, train_acc, val_loss_adj, val_acc_adj,
            lr_head, lr_lora, secs, per_class_metrics, classes
        )
        metrics_f.flush()
        
        
        base_metrics = {
            "train_loss": train_loss,
            "train_acc":  train_acc,
            "val_loss_eff": val_loss_adj,   # effective = LA (+mask)
            "val_acc_eff":  val_acc_adj,
            "val_loss_raw": val_loss_raw,
            "val_acc_raw":  val_acc_raw,
            "macro_f1":     float(macro_f1 if macro_f1==macro_f1 else 0.0),
            "tau_eff":      tau_eff,
            "label_smoothing": label_smoothing,
            "epoch_secs":   secs,
        }
        lrs = {
            "head": lr_head,
            "head_w": _find_group_lr(optimizer, set(head_weight_params)) or 0.0,
            "head_s": _find_group_lr(optimizer, set(head_scale_params)) or 0.0,
            "lora": _find_group_lr(optimizer, set(lora_params) if isinstance(lora_params, list) else set()) or 0.0,
            "backbone": lr_backbone,
        }

        log_to_wandb(wandb_mod, wb_run, epoch, base_metrics, ema_metrics, lrs)


        # ---- Save best (student vs EMA) ----
        metric_now = (val_acc_adj if save_by == "acc" else (macro_f1 if macro_f1==macro_f1 else 0.0))
        metric_src = "student"
        if ema_metrics is not None:
            ema_now = (ema_metrics["val_eff/acc"] if save_by == "acc" else ema_metrics["macro_f1"])
            if ema_now >= metric_now:
                metric_now = ema_now
                metric_src = "ema"

        if metric_now > best_metric:
            best_metric = metric_now
            to_save = (ema_model.module.state_dict() if (ema_model is not None and metric_src == "ema")
                       else model.state_dict())
            torch.save({
                "model": to_save,
                "args": vars(args),
                "classes": classes,
                "val_acc_eff": (ema_metrics["val_eff/acc"] if (ema_metrics and metric_src=="ema") else val_acc_adj),
                "macro_f1": float((ema_metrics["macro_f1"] if (ema_metrics and metric_src=="ema")
                                else (macro_f1 if macro_f1==macro_f1 else 0.0))),
                "best_metric_name": save_by,
                "best_metric_value": best_metric,
                "ema": bool(ema_model is not None and metric_src=="ema"),
            }, best_path)
            print(f"  ✓ Saved new best ({save_by}={best_metric:.3f}, from={metric_src}) to {best_path}")

        # Early stopping (macro‑F1) — off in your acc‑first runs unless you enable it
        if (not early_stop_off) and (macro_f1 is not None) and (macro_f1==macro_f1):
            if MacroF1EarlyStopping(patience=20, min_delta=0.002)(macro_f1):
                break

    if metrics_f is not None:
        metrics_f.close()
    try:
        if 'wb_run' in locals() and wb_run is not None:
            wb_run.finish()
    except Exception:
        pass

    print(f"\nDone. Best {save_by}={best_metric:.3f}")
    print(f"Checkpoint: {best_path}")


if __name__ == "__main__":
    main()
