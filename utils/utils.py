# utils.py
from pathlib import Path
from datetime import datetime
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.distributed as dist
import os
from typing import List, Sequence, Union, Tuple, Dict, Optional,Callable,Iterable, Any
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support,f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch.nn.functional as F

import math
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy




# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------
# Sampling / imbalance
# -----------------------------
def make_weighted_sampler(df, label_col: str):
    counts = df[label_col].value_counts().to_dict()
    weights = df[label_col].map(lambda c: 1.0 / counts[c]).values
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True
    )

# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(logits: torch.Tensor, y: torch.Tensor):
    pred = logits.argmax(1)
    correct = (pred == y).sum().item()
    total = y.size(0)
    return correct, total

def per_class_counts(loader, model, device, num_classes: int):
    model.eval()
    counts = np.zeros(num_classes, dtype=np.int64)
    with torch.inference_mode():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            preds = logits.argmax(1).cpu().numpy()
            for p in preds:
                counts[p] += 1
    return counts

# -----------------------------
# NEW: Per-Class Metrics
# -----------------------------

def compute_per_class_metrics(all_preds, all_targets, classes):
    if len(all_preds) == 0 or len(all_targets) == 0:
        return {}, np.zeros(len(classes)), np.zeros(len(classes))
    p = np.asarray(all_preds.cpu() if hasattr(all_preds, "cpu") else all_preds)
    t = np.asarray(all_targets.cpu() if hasattr(all_targets, "cpu") else all_targets)

    overall_acc = (p == t).mean()
    # macro by average=None, then simple means
    prec, rec, f1, support = precision_recall_fscore_support(
        t, p, labels=np.arange(len(classes)), average=None, zero_division=0
    )
    # per-class accuracy = recall for single label multiclass
    per_class_acc = rec.copy()

    metrics = {
        "overall_acc": overall_acc,
        "per_class_precision": prec,
        "per_class_recall": rec,
        "per_class_f1": f1,
        "per_class_support": support.astype(float),
        "per_class_acc": per_class_acc,
        "macro_precision": prec.mean(),
        "macro_recall": rec.mean(),
        "macro_f1": f1.mean(),
        "macro_acc": per_class_acc.mean(),
    }
    return metrics, per_class_acc, support


def print_per_class_summary(per_class_metrics, classes):
    """Print a nice summary of per-class performance."""
    if not per_class_metrics:
        return
    
    print(f"  Macro Acc: {per_class_metrics['macro_acc']:.3f}  "
          f"Macro F1: {per_class_metrics['macro_f1']:.3f}")
    
    # Print worst and best performing classes
    sorted_classes = sorted(enumerate(classes), 
                           key=lambda x: per_class_metrics['per_class_acc'][x[0]], 
                           reverse=True)
    
    print(f"  Best classes: ", end="")
    for i, (idx, cls) in enumerate(sorted_classes[:3]):
        acc = per_class_metrics['per_class_acc'][idx]
        support = int(per_class_metrics['per_class_support'][idx])
        print(f"{cls}({acc:.3f}, n={support})", end=", " if i < 2 else "\n")
    
    print(f"  Worst classes: ", end="")
    for i, (idx, cls) in enumerate(sorted_classes[-3:]):
        acc = per_class_metrics['per_class_acc'][idx]
        support = int(per_class_metrics['per_class_support'][idx])
        print(f"{cls}({acc:.3f}, n={support})", end=", " if i < 2 else "\n")

# -----------------------------
# CSV metrics logger 
# -----------------------------

def init_metrics_logger(out_dir, filename: str = "metrics.csv", classes=None):
    """Enhanced metrics logger with per-class metrics."""
    path = Path(out_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if new:
        header = [
            "run_id", "epoch", "train_loss", "train_acc",
            "val_loss", "val_acc", "lr_head", "lr_lora",
            "secs", "timestamp_utc"
        ]
        
        if classes:
            header.extend(["macro_acc", "macro_precision", "macro_recall", "macro_f1"])
            header.extend([f"acc_{cls.replace(' ', '_').replace('-', '_')}" for cls in classes])
        
        w.writerow(header)
        f.flush()
    return f, w


def log_metrics(writer, run_id, epoch, train_loss, train_acc, val_loss, val_acc, 
                        lr_head, lr_lora, secs, per_class_metrics=None, classes=None):
    """Enhanced metrics logging with per-class accuracy."""
    row = [
        run_id, epoch,
        f"{train_loss:.6f}", f"{train_acc:.6f}",
        f"{val_loss:.6f}", f"{val_acc:.6f}",
        f"{lr_head:.8f}", f"{lr_lora:.8f}",
        f"{secs:.3f}", datetime.utcnow().isoformat()
    ]
    
    if per_class_metrics and classes:
        # Add macro averages
        row.extend([
            f"{per_class_metrics['macro_acc']:.6f}",
            f"{per_class_metrics['macro_precision']:.6f}",
            f"{per_class_metrics['macro_recall']:.6f}",
            f"{per_class_metrics['macro_f1']:.6f}"
        ])
        
        # Add per-class accuracies
        row.extend([f"{acc:.6f}" for acc in per_class_metrics['per_class_acc']])
    
    writer.writerow(row)
    
    
    
    
# -----------------------------
# Class weights and priors
# -----------------------------

def class_counts_from_df(df, label_col: str, classes: Sequence[str]) -> np.ndarray:
    """
    Return counts aligned to your head order in `classes`.
    Works whether df[label_col] holds class names or integer indices.
    Missing classes get 0.
    """
    vc = df[label_col].value_counts()
    counts = np.zeros(len(classes), dtype=np.int64)
    for i, cls in enumerate(classes):
        if cls in vc.index:
            counts[i] = int(vc[cls])
        elif i in vc.index:
            counts[i] = int(vc[i])
        else:
            counts[i] = 0
    return counts

def class_counts_from_csv(csv_path: str, label_col: str, classes: Sequence[str]) -> np.ndarray:
    """
    Convenience loader if you prefer to compute weights without keeping df around.
    """
    import pandas as pd  # local import to avoid hard dependency at module import
    df = pd.read_csv(csv_path)
    return class_counts_from_df(df, label_col, classes)

def class_weight_tensor_from_counts(
    counts: np.ndarray,
    scheme: str = "inv_sqrt",
    normalize: Optional[str] = "mean",   # "mean", "sum", or None
    clamp: Optional[Tuple[float, float]] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build a weight tensor for CrossEntropyLoss from per class counts.
    schemes:
      - "inv": 1 / n_c
      - "inv_sqrt": 1 / sqrt(n_c)   [default, usually more stable]
      - "none": all ones
    Any class with count 0 receives weight 0. This ignores it in the loss.
    """
    counts = np.asarray(counts, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        if scheme == "inv":
            w = np.where(counts > 0, 1.0 / counts, 0.0)
        elif scheme == "inv_sqrt":
            w = np.where(counts > 0, 1.0 / np.sqrt(counts), 0.0)
        elif scheme == "none":
            w = np.ones_like(counts, dtype=np.float64)
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

    # normalize
    if normalize == "mean":
        m = w[w > 0].mean() if np.any(w > 0) else 1.0
        if m > 0:
            w = w / m
    elif normalize == "sum":
        s = w.sum()
        if s > 0:
            w = w / s

    # clamp if requested
    if clamp is not None:
        lo, hi = clamp
        w = np.clip(w, lo, hi)

    return torch.tensor(w, dtype=dtype, device=device)

def prior_logits_from_counts(counts: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Log priors for bias init: log(n_c / N).
    """
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    p = counts / max(total, eps)
    return np.log(p + eps)

def init_classifier_bias_from_counts(head_module: torch.nn.Module, counts: np.ndarray) -> None:
    """
    Copy prior logits into a linear head bias in place.
    Does nothing if no bias.
    """
    if hasattr(head_module, "bias") and head_module.bias is not None:
        b = prior_logits_from_counts(counts)
        with torch.no_grad():
            head_module.bias.copy_(
                torch.from_numpy(b)
                .to(head_module.bias.device)
                .to(head_module.bias.dtype)
            )


def maybe_init_wandb(args, classes, run_id: str, out_dir: str):
    """
    Enable W&B if env WANDB=1. Optional envs:
      WANDB_PROJECT, WANDB_RUN_NAME, WANDB_MODE=offline
    Returns (wandb_module, run) or (None, None) if disabled or unavailable.
    """
    if os.environ.get("WANDB", "0") != "1":
        return None, None
    try:
        import wandb as wb
    except Exception as e:
        print(f"WandB not available: {e}")
        return None, None
    project = os.environ.get("WANDB_PROJECT", "cellpt")
    name = os.environ.get("WANDB_RUN_NAME", f"{Path(out_dir).name}-{run_id}")
    cfg = dict(vars(args))
    cfg.update({"num_classes": len(classes), "classes": classes})
    run = wb.init(project=project, name=name, config=cfg)
    print(f"WandB logging enabled: project={project}, run={name}")
    return wb, run


def confusion_matrix_df(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: List[str]
) -> pd.DataFrame:
    """
    Return a pandas DataFrame confusion matrix with labeled rows and columns.
    Rows are true classes, columns are predicted classes.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    idx = [f"true:{c}" for c in class_names]
    cols = [f"pred:{c}" for c in class_names]
    return pd.DataFrame(cm, index=idx, columns=cols)

def _name_to_index(name: Union[str, int], class_names: List[str]) -> int:
    if isinstance(name, int):
        return name
    # exact, case insensitive
    lower = [c.lower() for c in class_names]
    if name.lower() in lower:
        return lower.index(name.lower())
    # fallback: first contains match
    for i, c in enumerate(lower):
        if name.lower() in c:
            return i
    raise ValueError(f"Class '{name}' not found in class_names")

def print_confusion_slice(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: List[str],
    target_classes: List[Union[str, int]],
    topk: int = 3
) -> None:
    """
    Print precision, recall, F1, support and top-k confusion destinations
    for the selected classes. Also shows True percent, Pred percent, and their ratio.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    K = len(class_names)
    labels = np.arange(K)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # per-class PRF
    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    # distribution
    true_counts = cm.sum(axis=1)
    pred_counts = cm.sum(axis=0)
    total_true = true_counts.sum()
    total_pred = pred_counts.sum()
    print("\n[Confusion slice]")
    for name in target_classes:
        i = _name_to_index(name, class_names)
        row = cm[i, :].copy()
        row[i] = 0
        if row.sum() > 0:
            top_idx = np.argsort(row)[::-1][:topk]
        else:
            top_idx = []
        tpct = 100.0 * (true_counts[i] / total_true) if total_true > 0 else 0.0
        ppct = 100.0 * (pred_counts[i] / total_pred) if total_pred > 0 else 0.0
        ratio = (ppct / tpct) if tpct > 0 else 0.0
        print(f"- {class_names[i]}  P={prec[i]:.3f}  R={rec[i]:.3f}  F1={f1[i]:.3f}  support={int(supp[i])}  True%={tpct:.1f}  Pred%={ppct:.1f}  Ratio={ratio:.2f}")
        if len(top_idx) > 0:
            for j in top_idx:
                cnt = int(cm[i, j])
                frac = (cnt / max(1, true_counts[i]))
                print(f"    confuses to {class_names[j]}: {cnt}  ({frac:.2%} of true {class_names[i]})")
                
                
                
# -----------------------------
# Confusion matrix save + print
# -----------------------------

def _cm_norm_key(norm: Optional[str]) -> Optional[str]:
    if norm is None:
        return None
    s = str(norm).strip().lower()
    if s in {"true", "pred", "all"}:
        return s
    if s in {"none", "false", "0", ""}:
        return None
    raise ValueError(f"Unknown normalization: {norm}")


def print_confusion_full(cm: np.ndarray, classes: List[str]) -> None:
    """
    Print the full confusion matrix with class headers.
    Uses integer format for counts and fixed 3 decimals for floats.
    """
    k = len(classes)
    # clamp names to keep width manageable in logs
    names = [c if len(c) <= 20 else c[:20] + "…" for c in classes]
    # choose width based on data type
    is_float = np.issubdtype(cm.dtype, np.floating)
    cellw = 8
    header = " " * (cellw + 1) + " ".join(f"{n:>{cellw}}" for n in names)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(header)
    for i in range(k):
        row = []
        for j in range(k):
            if is_float:
                row.append(f"{cm[i, j]:>{cellw}.3f}")
            else:
                row.append(f"{int(cm[i, j]):>{cellw}d}")
        print(f"{names[i]:>{cellw}} " + " ".join(row))


def _plot_confusion(
    cm: np.ndarray,
    classes: List[str],
    title: str,
    png_path: Path,
    annotate: bool = True,
    dpi: int = 160
) -> None:
    """
    Render and save a confusion matrix heatmap to png_path.
    """
    k = len(classes)
    xt = [c if len(c) <= 20 else c[:20] + "…" for c in classes]
    yt = [c if len(c) <= 20 else c[:20] + "…" for c in classes]

    # scale figure size modestly with number of classes
    fig_w = max(6, min(18, 0.5 * k))
    fig_h = fig_w
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(k),
        yticks=np.arange(k),
        xticklabels=xt,
        yticklabels=yt,
        ylabel="True class",
        xlabel="Predicted class",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if annotate:
        is_float = np.issubdtype(cm.dtype, np.floating)
        fmt = ".2f" if is_float else "d"
        thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
        for i in range(k):
            for j in range(k):
                val = format(cm[i, j], fmt)
                ax.text(
                    j, i, val,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8
                )

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrices(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: List[str],
    out_dir: Union[str, Path],
    epoch: Optional[int] = None,
    norms: Sequence[str] = ("none", "true", "pred", "all"),
    print_table_for: Optional[str] = "true",
    annotate: bool = True
) -> Dict[str, Dict[str, Path]]:
    """
    Compute confusion matrices for multiple normalizations and save both CSV and PNG.

    Args:
        y_true, y_pred: arrays or tensors of integer class ids aligned to class_names.
        class_names: list of class labels in index order.
        out_dir: directory to write files.
        epoch: if provided, file names include epoch number.
        norms: which normalizations to compute. Any of {"none","true","pred","all"}.
        print_table_for: if one of the norms, print a full table for that norm.
        annotate: add numbers in the heatmap cells.

    Returns:
        Dict mapping norm -> {"csv": Path, "png": Path}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.arange(len(class_names))
    out_dir = Path(out_dir)

    results: Dict[str, Dict[str, Path]] = {}
    for raw_norm in norms:
        norm = _cm_norm_key(raw_norm)
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=norm)

        # paths
        tag = (norm if norm is not None else "none")
        ep = f"_epoch{epoch}" if epoch is not None else ""
        csv_path = out_dir / f"confusion{ep}_{tag}.csv"
        png_path = out_dir / f"confusion{ep}_{tag}.png"

        # CSV
        row_labels = [f"true:{c}" for c in class_names]
        col_labels = [f"pred:{c}" for c in class_names]
        pd.DataFrame(cm, index=row_labels, columns=col_labels).to_csv(
            csv_path, float_format="%.6f"
        )

        # PNG
        title = f"Confusion matrix{f' epoch {epoch}' if epoch is not None else ''} (normalize={tag})"
        _plot_confusion(cm, class_names, title, png_path, annotate=annotate)

        results[tag] = {"csv": csv_path, "png": png_path}

        # Optional console print for one selected norm
        if _cm_norm_key(print_table_for) == norm:
            print_confusion_full(cm, class_names)

    return results




def worker_init_fn(seed: int) -> Callable[[int], None]:
    """Create a deterministic worker init function for DataLoader."""
    def _init(worker_id: int):
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)
    return _init



def apply_slide_mask(logits: torch.Tensor,
                     slide_ids: List[str] | None,
                     mask_table: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Mask out logits for classes not observed on a slide. Unknown slides are left unmasked."""
    if slide_ids is None:
        return logits
    out = logits.clone()
    neg_inf = torch.finfo(out.dtype).min
    for i, sid in enumerate(slide_ids):
        m = mask_table.get(sid, None)
        if m is None:
            continue
        m = m.to(device=out.device)
        out[i, ~m] = neg_inf
    return out


def evaluate_with_mask(model: torch.nn.Module,
                       val_loader,
                       device: torch.device,
                       amp_dtype: torch.dtype,
                       adj: torch.Tensor | None,
                       use_slide_mask: bool,
                       allowed_bitmask: Dict[str, torch.Tensor],
                       classes: List[str]) -> dict:
    """
    Validation with optional logit adjustment and slide masking.
    Returns dict with effective and raw metrics plus preds and targets.
    """

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

            # effective path: LA then optional slide mask
            logits_eff = logits if adj is None else logits - adj.to(dtype=logits.dtype)
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


def log_to_wandb(wandb_mod, wb_run, epoch: int, base_metrics: dict,
                 ema_metrics: dict | None, lrs: dict, extra: dict | None = None) -> None:
    """Centralized W&B logging."""
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


class MacroF1EarlyStopping:
    """Track macro F1 across epochs and signal stop when patience is exceeded."""
    def __init__(self, patience: int = 8, min_delta: float = 0.002):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -1.0
        self.counter = 0

    def step(self, macro_f1: float) -> bool:
        """Return True if training should stop."""
        if macro_f1 > self.best_score + self.min_delta:
            self.best_score = macro_f1
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def detailed_prediction_analysis(all_preds: torch.Tensor,
                                 all_targets: torch.Tensor,
                                 classes: List[str],
                                 epoch: int,
                                 interval: int = 10) -> None:
    """Every interval epochs, print predicted share vs true share per class."""
    if all_preds.numel() == 0 or (epoch % interval != 0):
        return
    pred_counts = np.bincount(all_preds.cpu().numpy(), minlength=len(classes))
    true_counts = np.bincount(all_targets.cpu().numpy(), minlength=len(classes))
    total_pred, total_true = pred_counts.sum(), true_counts.sum()
    print("\nPrediction breakdown:")
    print(f"{'Class':<25} {'True %':<8} {'Pred %':<8} {'Ratio':<8} Status")
    for i, cls in enumerate(classes):
        tp = (true_counts[i] / total_true * 100.0) if total_true > 0 else 0.0
        pp = (pred_counts[i] / total_pred * 100.0) if total_pred > 0 else 0.0
        ratio = (pp / tp) if tp > 0 else 0.0
        status = "OK" if 0.7 <= ratio <= 1.4 else ("High" if ratio > 1.4 else "Low")
        print(f"{cls:<25} {tp:<8.1f} {pp:<8.1f} {ratio:<8.2f} {status}")


def build_allowed_bitmask(train_df, classes: List[str], label_col: str, slide_col: str = "slide_id") -> Dict[str, torch.Tensor]:
    """
    Build slide to class bitmask map. A True entry marks classes that appear on the slide in train.
    """
    slide_to_allowed = (
        train_df.groupby(slide_col)[label_col]
        .apply(lambda s: sorted(s.unique().tolist()))
        .to_dict()
    )
    allowed_bitmask: Dict[str, torch.Tensor] = {}
    class_to_idx_local = {c: i for i, c in enumerate(classes)}
    for sid, cls_list in slide_to_allowed.items():
        m = np.zeros(len(classes), dtype=bool)
        for c in cls_list:
            idx = class_to_idx_local.get(c, None)
            if idx is not None:
                m[idx] = True
        allowed_bitmask[sid] = torch.from_numpy(m)
    return allowed_bitmask


def build_allowed_bitmask_from_meta_json(
    classes: List[str],
    *,
    level: str = "fine",                 # "fine" or "coarse"
    subset: str = "subset_all",
    json_path: str = "/hpc/group/jilab/rz179/subset_per_slide_two_eval_splits_full/metadata.json",
) -> Dict[str, torch.Tensor]:
    """
    Build slide -> class bitmask from metadata.json with structure like:
      splits -> <subset> -> per_slide -> <slide_id> -> { "<level>": { "counts": {...} } }

    Falls back to:
      experiment -> splits -> <subset> -> per_slide
      and then top-level per_slide (if present)

    Returns dict[slide_id] -> BoolTensor[num_classes].
    Use only at evaluation.
    """
    import json

    with open(json_path, "r") as f:
        meta = json.load(f)

    # primary: top-level splits
    per_slide = (
        meta.get("splits", {})
            .get(subset, {})
            .get("per_slide", None)
    )

    # fallback 1: experiment -> splits
    if per_slide is None:
        per_slide = (
            meta.get("experiment", {})
                .get("splits", {})
                .get(subset, {})
                .get("per_slide", None)
        )

    # fallback 2: top-level per_slide
    if per_slide is None:
        per_slide = meta.get("per_slide", None)

    if not isinstance(per_slide, dict):
        have_top = list(meta.keys())
        have_splits = list(meta.get("splits", {}).keys()) if isinstance(meta.get("splits"), dict) else []
        have_exp_splits = list(meta.get("experiment", {}).get("splits", {}).keys()) \
            if isinstance(meta.get("experiment", {}), dict) and isinstance(meta["experiment"].get("splits"), dict) else []
        raise ValueError(
            "[mask] Could not locate 'per_slide'. "
            f"Checked splits->{subset}->per_slide, experiment->splits->{subset}->per_slide, and top-level. "
            f"Top-level keys: {have_top}. splits keys: {have_splits}. experiment.splits keys: {have_exp_splits}."
        )

    class_to_idx = {c: i for i, c in enumerate(classes)}
    masks: Dict[str, torch.Tensor] = {}
    missing_level = 0
    missing_counts = 0
    unknown_labels = 0

    for sid, rec in per_slide.items():
        sect = rec.get(level, None)
        if not isinstance(sect, dict):
            missing_level += 1
            continue
        counts = sect.get("counts", {})
        if not isinstance(counts, dict):
            missing_counts += 1
            continue

        m = np.zeros(len(classes), dtype=bool)
        for name, cnt in counts.items():
            if int(cnt) <= 0:
                continue
            idx = class_to_idx.get(name)
            if idx is None:
                unknown_labels += 1
                continue
            m[idx] = True
        masks[str(sid)] = torch.from_numpy(m)

    print(
        f"[mask] Built {len(masks)} slide masks from {json_path} "
        f"(level='{level}', subset='{subset}'). "
        f"slides_missing_level={missing_level} slides_missing_counts={missing_counts} unknown_labels={unknown_labels}"
    )
    return masks





def setup_mixup(args, num_classes: int) -> Tuple[bool, Any | None, Any | None, int]:
    """Return (enabled, mixup_fn, soft_ce, mixup_off_k)."""
    mixup_alpha  = float(getattr(args, "mixup_alpha", 0.0))
    cutmix_alpha = float(getattr(args, "cutmix_alpha", 0.0))
    mixup_prob   = float(getattr(args, "mixup_prob", 1.0))
    mixup_switch = float(getattr(args, "mixup_switch_prob", 0.0))
    mixup_mode   = str(getattr(args, "mixup_mode", "batch"))
    mixup_off_k  = int(getattr(args, "mixup_off_epochs", 10))

    enabled = (mixup_alpha > 0.0) or (cutmix_alpha > 0.0)
    if not enabled:
        print("Mixup/CutMix disabled.")
        return False, None, None, mixup_off_k

    mixup_fn = Mixup(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        prob=mixup_prob,
        switch_prob=mixup_switch,
        mode=mixup_mode,
        label_smoothing=0.0,
        num_classes=num_classes,
    )
    soft_ce = SoftTargetCrossEntropy()
    print(f"Mixup/CutMix enabled: mixup_alpha={mixup_alpha} cutmix_alpha={cutmix_alpha}")
    return True, mixup_fn, soft_ce, mixup_off_k


def create_scheduler(args, optimizer) -> torch.optim.lr_scheduler._LRScheduler | None:
    """Cosine with warmup, else None."""
    if not bool(getattr(args, "cosine", False)):
        return None

    total_epochs = int(getattr(args, "epochs", 100))
    warmup = max(0, min(int(getattr(args, "warmup_epochs", 5)), total_epochs - 1))

    def lr_lambda(epoch_idx: int):
        if epoch_idx < warmup:
            return float(epoch_idx + 1) / float(max(1, warmup))
        progress = (epoch_idx - warmup) / float(max(1, total_epochs - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def find_group_lr(optimizer: torch.optim.Optimizer, param_set: Iterable[torch.nn.Parameter]) -> float | None:
    """Return lr of the first param group that contains any param from param_set (by identity)."""
    target_ids = {id(p) for p in param_set}
    for g in optimizer.param_groups:
        if any(id(p) in target_ids for p in g["params"]):
            return float(g.get("lr", 0.0))
    return None


def build_optimizer(model: torch.nn.Module,
                    args,
                    lora_params,
                    norms,
                    head_weight_params,
                    head_scale_params) -> torch.optim.Optimizer:
    """AdamW with grouped LRs. Ensures groups are disjoint by identity."""
    unfreeze = bool(getattr(args, "unfreeze_backbone", False))

    # helpers
    def uniq(params):
        seen, out = set(), []
        for p in params:
            if p is None: continue
            pid = id(p)
            if pid not in seen:
                out.append(p); seen.add(pid)
        return out

    # collect sets by identity
    all_trainable = [p for p in model.parameters() if p.requires_grad]

    head_w_ids  = {id(p) for p in uniq(head_weight_params)}
    head_s_ids  = {id(p) for p in uniq(head_scale_params)}
    lora_ids    = {id(p) for p in uniq(lora_params)}
    norm_ids    = {id(p) for p in uniq(norms)}

    # detect extra heads and add them to head_w set if present
    if hasattr(model, "coarse_head"):
        for p in model.coarse_head.parameters():
            head_w_ids.add(id(p))
    if hasattr(model, "scale_gate"):
        for p in model.scale_gate.parameters():
            head_w_ids.add(id(p))

    used = set()

    def take(ids_set, base_iter=None):
        base = all_trainable if base_iter is None else base_iter
        picked = []
        for p in base:
            pid = id(p)
            if pid in ids_set and pid not in used:
                picked.append(p); used.add(pid)
        return picked

    # build disjoint groups in a stable order
    groups = []

    if unfreeze:
        # backbone = everything trainable not explicitly assigned to heads/lora/norms
        # heads come first so backbone won't grab them
        group_head_w = take(head_w_ids)
        group_head_s = take(head_s_ids)
        # no lora group typically when unfreezing ViT, but still subtract them from backbone if present
        group_lora   = take(lora_ids)
        group_norms  = take(norm_ids)  # if any norms require grad
        group_backbone = [p for p in all_trainable if id(p) not in used and (used.add(id(p)) or True)]

        if group_backbone:
            groups.append({"params": group_backbone, "lr": float(getattr(args, "lr_backbone", 0.0)), "weight_decay": 5e-2})
        if group_head_w:
            groups.append({"params": group_head_w, "lr": float(getattr(args, "lr_head", 1e-3)), "weight_decay": 1e-4})
        if group_head_s:
            groups.append({"params": group_head_s, "lr": float(getattr(args, "lr_head", 1e-3)), "weight_decay": 0.0})
        if group_lora:
            groups.append({"params": group_lora, "lr": float(getattr(args, "lr_lora", 1e-4)), "weight_decay": 0.0})
        if group_norms:
            groups.append({"params": group_norms, "lr": float(getattr(args, "lr_backbone", 0.0)), "weight_decay": 0.0})
    else:
        group_head_w = take(head_w_ids)
        group_head_s = take(head_s_ids)
        group_lora   = take(lora_ids)
        group_norms  = take(norm_ids)

        if group_lora:
            groups.append({"params": group_lora,   "lr": float(getattr(args, "lr_lora", 1e-4)), "weight_decay": 0.0})
        if group_head_w:
            groups.append({"params": group_head_w, "lr": float(getattr(args, "lr_head", 1e-3)), "weight_decay": 1e-4})
        if group_head_s:
            groups.append({"params": group_head_s, "lr": float(getattr(args, "lr_head", 1e-3)), "weight_decay": 0.0})
        if group_norms:
            # no-decay for norms
            groups.append({"params": group_norms,  "lr": float(getattr(args, "lr_head", 1e-3))*0.2, "weight_decay": 0.0})

    # sanity: no overlaps, no unassigned
    assigned = {id(p) for g in groups for p in g["params"]}
    unassigned = [p for p in all_trainable if id(p) not in assigned]
    if unassigned:
        # should not happen, but fail fast with a clear message
        na = sum(p.numel() for p in unassigned)
        raise RuntimeError(f"Unassigned trainable params: {len(unassigned)} tensors ({na} elements). Check grouping logic.")

    # build optimizer
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
    return opt




def load_priors_from_csv(prior_csv: str | None,
                         label_col: str,
                         class_to_idx: Dict[str, int],
                         classes: List[str],
                         fallback_counts: np.ndarray) -> np.ndarray:
    """Return counts per class from a CSV of images if provided and valid, else the fallback."""
    if prior_csv is None or not str(prior_csv).strip():
        return fallback_counts
    try:
        cdf = pd.read_csv(prior_csv)
        cand_cols = ["img_path_uniform", "img_path"]
        col = next((c for c in cand_cols if c in cdf.columns), None)
        if col is None:
            raise RuntimeError("No image path column found in prior_csv.")
        cdf = cdf[(cdf[col].map(lambda p: Path(str(p)).is_file()))]
        cdf = cdf[cdf[label_col].isin(class_to_idx.keys())]
        vc = cdf[label_col].value_counts()
        out = np.array([int(vc.get(c, 0)) for c in classes], dtype=np.int64)
        print(f"Using priors from prior_csv={prior_csv}")
        return out
    except Exception as e:
        print(f"Warning: failed to read prior_csv ({e}); falling back to train counts.")
        return fallback_counts


def compute_logit_prior(counts_np: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return prior and log_prior tensors on device."""
    prior = torch.from_numpy(counts_np.astype(np.float64) / max(1, counts_np.sum()))
    prior = prior.clamp_min(1e-12).to(device=device, dtype=torch.float32)
    log_prior = torch.log(prior)
    return prior, log_prior


def tau_at_epoch(epoch: int, start: int, ramp: int, target: float) -> float:
    """Piecewise linear ramp from 0 to target starting at epoch start over ramp epochs."""
    if epoch < start:
        return 0.0
    if ramp <= 0:
        return float(target)
    prog = min(1.0, max(0.0, (epoch - start + 1) / float(max(1, ramp))))
    return float(target) * prog

def build_dataloaders(
    train_ds,
    val_ds,
    args,
    classes: List[str],
    label_col: str,
    worker_init_fn,
    smooth_on: bool = False,
    smooth_alpha: float = 0.5,
    epoch_budget: int = 0,
    balanced_last_k: int = 0,
    device: torch.device | None = None,
) -> Dict[str, DataLoader | None]:
    """
    Create train and val DataLoaders:
      train_shuffle    uniform shuffle
      train_weighted   1/n_c sampling if args.sampler == 'weighted'
      train_smooth     power smoothing if smooth_on is True
      train_balanced   1/n_c for last K epochs if balanced_last_k > 0
      val              deterministic val
    """
    pin = bool(device is not None and getattr(device, "type", "cpu") == "cuda")
    pw_flag = bool(getattr(args, "num_workers", 0) > 0 and pin)

    # uniform shuffle
    train_loader_shuffle = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=pin,
        persistent_workers=pw_flag,
        drop_last=True,
    )

    # weighted 1/n_c if requested
    train_loader_weighted = None
    if getattr(args, "sampler", "uniform") == "weighted":
        vc = train_ds.df[label_col].value_counts()
        n = np.array([int(vc.get(c, 0)) for c in classes], dtype=np.float64)
        w_per_class = 1.0 / np.clip(n, 1.0, None)
        w_map = {c: w_per_class[i] for i, c in enumerate(classes)}
        w = torch.tensor([w_map[v] for v in train_ds.df[label_col].tolist()], dtype=torch.float32)
        sampler_w = WeightedRandomSampler(w, num_samples=len(train_ds.df), replacement=True)
        train_loader_weighted = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler_w,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=worker_init_fn,
            pin_memory=pin,
            persistent_workers=pw_flag,
            drop_last=True,
        )
        print("[sampler] Using WEIGHTED sampler 1/n_c.")

    # val
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=pin,
        persistent_workers=pw_flag,
    )

    # smoothing
    train_loader_smooth = None
    if smooth_on:
        vc = train_ds.df[label_col].value_counts()
        n = np.array([int(vc.get(c, 0)) for c in classes], dtype=np.float64).clip(1.0, None)
        w_per_class = n ** (smooth_alpha - 1.0)
        w_map = {c: w_per_class[i] for i, c in enumerate(classes)}
        w = torch.tensor([w_map[cell_type] for cell_type in train_ds.df[label_col]], dtype=torch.float32)
        num_samples = int(epoch_budget) if int(epoch_budget) > 0 else len(train_ds)
        implied = n * w_per_class
        implied = implied / implied.sum()
        head_i = int(np.argmax(n))
        tail_i = int(np.argmin(n))
        print(
            f"Smoothing alpha={smooth_alpha:.2f} implied share  "
            f"head {classes[head_i]}={implied[head_i]:.4f}  "
            f"tail {classes[tail_i]}={implied[tail_i]:.4f}  "
            f"budget={'full' if epoch_budget==0 else num_samples}"
        )
        smooth_sampler = WeightedRandomSampler(w, num_samples=num_samples, replacement=True)
        train_loader_smooth = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=smooth_sampler,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=worker_init_fn,
            pin_memory=pin,
            persistent_workers=pw_flag,
            drop_last=True,
        )

    # balanced for last K epochs
    train_loader_balanced = None
    if int(balanced_last_k) > 0:
        vc = train_ds.df[label_col].value_counts()
        n = np.array([int(vc.get(c, 0)) for c in classes], dtype=np.float64)
        w_per_class = 1.0 / np.clip(n, 1.0, None)
        w_map = {c: w_per_class[i] for i, c in enumerate(classes)}
        w = torch.tensor([w_map[v] for v in train_ds.df[label_col].tolist()], dtype=torch.float32)
        sampler_bal = WeightedRandomSampler(w, num_samples=len(train_ds.df), replacement=True)
        train_loader_balanced = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler_bal,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=worker_init_fn,
            pin_memory=pin,
            persistent_workers=pw_flag,
            drop_last=True,
        )
        print(f"[sampler] Balanced sampler will be used for the last {int(balanced_last_k)} epochs.")

    return {
        "train_shuffle": train_loader_shuffle,
        "train_weighted": train_loader_weighted,
        "train_smooth": train_loader_smooth,
        "train_balanced": train_loader_balanced,
        "val": val_loader,
    }


def build_loss_fn(args, cls_counts: np.ndarray, device: torch.device):
    from utils.losses import CBFocalLoss, LDAMLoss
    if args.loss_type == "cb_focal":
        loss_obj = CBFocalLoss(cls_counts.astype(int), beta=args.cb_beta, gamma=args.focal_gamma).to(device)
        return lambda logits, y, epoch=None: loss_obj(logits, y)
    if args.loss_type == "ldam_drw":
        loss_obj = LDAMLoss(cls_counts.astype(int), max_m=args.ldam_max_m, s=args.ldam_s,
                            drw=True, beta=args.cb_beta, drw_start=args.drw_start_epoch).to(device)
        return lambda logits, y, epoch=None: loss_obj(logits, y, epoch=epoch)
    # default CE with label smoothing handled in the loop
    return None



def build_taxonomy_from_meta_json(
    classes_fine: List[str],
    *,
    subset: str = "subset_all",
    json_path: str = "/hpc/group/jilab/rz179/subset_per_slide_two_eval_splits_full/metadata.json",
    device: torch.device | None = None,
) -> tuple[list[str], dict[str, int], torch.Tensor, dict[str, str]]:
    """
    Returns:
      coarse_classes: list of coarse names in index order
      coarse_to_idx: dict coarse name -> index
      fine_to_coarse_idx: LongTensor [K] mapping each fine index to a coarse index
      fine_to_coarse_name: dict fine name -> coarse name
    """
    import json
    from pathlib import Path

    # Try loading from dedicated mapping file first
    mapping_path = Path("/hpc/group/jilab/rz179/cellpt/combined/fine_to_coarse_map.json")
    fine_to_coarse_name: dict[str, str] = {}
    
    if mapping_path.exists():
        with open(mapping_path, "r") as f:
            fine_to_coarse_name = json.load(f)
        print(f"[coarse] Loaded taxonomy from {mapping_path}")
    else:
        # Fallback: try metadata.json
        with open(json_path, "r") as f:
            meta = json.load(f)
        
        def _get_taxonomy(root):
            if not isinstance(root, dict):
                return None
            tx = root.get("taxonomy", {})
            if isinstance(tx, dict):
                return tx
            return None
        
        tx = None
        tx = tx or _get_taxonomy(meta)
        tx = tx or _get_taxonomy(meta.get("splits", {}).get(subset, {}))
        tx = tx or _get_taxonomy(meta.get("experiment", {}).get("splits", {}).get(subset, {}))
        
        if tx:
            if isinstance(tx.get("fine_to_coarse"), dict):
                fine_to_coarse_name = dict(tx["fine_to_coarse"])
            elif isinstance(tx.get("coarse_members"), dict):
                for coarse_name, fine_list in tx["coarse_members"].items():
                    for fn in fine_list:
                        fine_to_coarse_name[fn] = coarse_name
        
        if not fine_to_coarse_name:
            raise ValueError(
                f"[coarse] No mapping found. Expected either:\n"
                f"  1. {mapping_path}\n"
                f"  2. taxonomy.fine_to_coarse in {json_path}"
            )
    
    # Validate: check which fine classes have mappings
    coarse_classes = sorted({fine_to_coarse_name.get(fn, "") for fn in classes_fine if fn in fine_to_coarse_name})
    missing = [fn for fn in classes_fine if fn not in fine_to_coarse_name]
    if missing:
        print(f"[coarse] WARNING: {len(missing)} fine classes have no coarse mapping: {missing}")
    
    coarse_to_idx = {c: i for i, c in enumerate(coarse_classes)}
    idx_list = []
    for fn in classes_fine:
        cn = fine_to_coarse_name.get(fn, None)
        if cn is None:
            idx_list.append(-1)  # unmapped
        else:
            idx_list.append(coarse_to_idx[cn])
    
    fine_to_coarse_idx = torch.tensor(idx_list, dtype=torch.long)
    if device is not None:
        fine_to_coarse_idx = fine_to_coarse_idx.to(device)
    
    print(f"[coarse] {len(coarse_classes)} coarse groups: {coarse_classes}")
    return coarse_classes, coarse_to_idx, fine_to_coarse_idx, fine_to_coarse_name


def build_coarse_logpriors_from_meta_json(
    coarse_classes: List[str],
    *,
    subset: str = "subset_all",
    json_path: str = "/hpc/group/jilab/rz179/subset_per_slide_two_eval_splits_full/metadata.json",
    alpha: float = 5.0,
    device: torch.device | None = None,
    # NEW: if provided, aggregate fine counts into these coarse groups
    fine_to_coarse_name: dict[str, str] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Builds per-slide coarse priors with add-alpha smoothing and returns log probabilities.

    If `fine_to_coarse_name` is provided, we aggregate per-slide `fine.counts`
    into your desired coarse groups (recommended when your coarse taxonomy differs
    from metadata.json). Otherwise we fall back to metadata's `coarse.counts`.

    Returns:
      slide_id -> FloatTensor [G] of log probabilities matching `coarse_classes` order.
    """
    import json

    with open(json_path, "r") as f:
        meta = json.load(f)

    per_slide = (
        meta.get("splits", {})
            .get(subset, {})
            .get("per_slide", None)
    )
    if not isinstance(per_slide, dict):
        per_slide = (
            meta.get("experiment", {})
                .get("splits", {})
                .get(subset, {})
                .get("per_slide", None)
        )
    if not isinstance(per_slide, dict):
        raise ValueError("[coarse] Could not locate splits -> subset -> per_slide in metadata.json")

    G = len(coarse_classes)
    c2i = {c: i for i, c in enumerate(coarse_classes)}
    out: dict[str, torch.Tensor] = {}

    use_aggregate = fine_to_coarse_name is not None
    unknown_fine = 0
    missing_coarse = 0

    for sid, rec in per_slide.items():
        v = torch.full((G,), fill_value=alpha, dtype=torch.float32)

        if use_aggregate:
            # aggregate fine.counts -> your coarse groups
            fcounts = rec.get("fine", {}).get("counts", {})
            if isinstance(fcounts, dict):
                for fine_name, cnt in fcounts.items():
                    coarse_name = fine_to_coarse_name.get(fine_name)
                    if coarse_name is None:
                        unknown_fine += 1
                        continue
                    idx = c2i.get(coarse_name)
                    if idx is None:
                        missing_coarse += 1
                        continue
                    if int(cnt) > 0:
                        v[idx] += float(cnt)
        else:
            # FALLBACK: trust metadata's coarse-counts names
            ccounts = rec.get("coarse", {}).get("counts", {})
            if isinstance(ccounts, dict):
                for name, cnt in ccounts.items():
                    idx = c2i.get(name)
                    if idx is not None and int(cnt) > 0:
                        v[idx] += float(cnt)

        v = v / v.sum()
        v = v.clamp_min(1e-8).log()
        if device is not None:
            v = v.to(device)
        out[str(sid)] = v

    src = "fine.counts (aggregated)" if use_aggregate else "coarse.counts (metadata names)"
    print(f"[coarse] built log priors for {len(out)} slides from {src} with alpha={alpha}")
    if use_aggregate and (unknown_fine or missing_coarse):
        print(f"[coarse] note: ignored {unknown_fine} fine labels not in mapping and "
              f"{missing_coarse} mapped coarse names not in `coarse_classes`")
    return out



def map_fine_to_coarse_arrays(
    y_pred_fine: Union[np.ndarray, torch.Tensor],
    y_true_fine: Union[np.ndarray, torch.Tensor],
    fine_to_coarse_idx: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map fine class ids to coarse class ids using fine_to_coarse_idx.
    Filters any items whose mapping is -1 (unmapped).
    Returns (y_pred_coarse, y_true_coarse) as numpy int arrays.
    """
    if isinstance(y_pred_fine, torch.Tensor):
        y_pred_f = y_pred_fine.detach().cpu().numpy()
    else:
        y_pred_f = np.asarray(y_pred_fine)
    if isinstance(y_true_fine, torch.Tensor):
        y_true_f = y_true_fine.detach().cpu().numpy()
    else:
        y_true_f = np.asarray(y_true_fine)

    ftc = fine_to_coarse_idx.detach().cpu().numpy().astype(np.int64)  # [K]
    y_pred_c = ftc[y_pred_f]
    y_true_c = ftc[y_true_f]

    keep = (y_pred_c >= 0) & (y_true_c >= 0)
    return y_pred_c[keep], y_true_c[keep]


def compute_coarse_metrics_from_fine(
    y_pred_fine: Union[np.ndarray, torch.Tensor],
    y_true_fine: Union[np.ndarray, torch.Tensor],
    fine_to_coarse_idx: torch.Tensor,
    coarse_classes: List[str],
) -> Tuple[dict, np.ndarray]:
    """
    Compute per coarse class recall (row accuracy), support, and macro recall,
    plus the coarse confusion matrix (counts) from fine predictions and targets.
    """
    y_pred_c, y_true_c = map_fine_to_coarse_arrays(y_pred_fine, y_true_fine, fine_to_coarse_idx)
    labels = np.arange(len(coarse_classes))
    cm = confusion_matrix(y_true_c, y_pred_c, labels=labels, normalize=None)

    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_recall = np.where(cm.sum(axis=1) > 0, np.diag(cm) / cm.sum(axis=1), 0.0)
    support = cm.sum(axis=1).astype(np.int64)
    macro_recall = float(per_class_recall.mean()) if len(per_class_recall) > 0 else 0.0

    metrics = {
        "coarse_macro_recall": macro_recall,
        "coarse_per_class_recall": per_class_recall,
        "coarse_support": support,
        "coarse_overall_acc": float(np.trace(cm) / max(1, cm.sum())),
    }
    return metrics, cm


def save_coarse_confusions_from_fine(
    y_true_fine: Union[np.ndarray, torch.Tensor],
    y_pred_fine: Union[np.ndarray, torch.Tensor],
    coarse_classes: List[str],
    fine_to_coarse_idx: torch.Tensor,
    out_dir: Union[str, Path],
    epoch: Optional[int] = None,
    norms: Sequence[str] = ("none", "true", "pred", "all"),
    print_table_for: Optional[str] = "true",
    annotate: bool = True,
) -> Dict[str, Dict[str, Path]]:
    """
    Map fine ids to coarse ids and save coarse confusion CSV and PNG using your existing saver.
    """
    y_pred_c, y_true_c = map_fine_to_coarse_arrays(y_pred_fine, y_true_fine, fine_to_coarse_idx)
    return save_confusion_matrices(
        y_true=y_true_c, y_pred=y_pred_c, class_names=coarse_classes,
        out_dir=out_dir, epoch=epoch, norms=norms,
        print_table_for=print_table_for, annotate=annotate
    )
