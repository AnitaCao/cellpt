# utils.py
from pathlib import Path
from datetime import datetime
import csv
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
import torch.distributed as dist
from typing import Optional, Sequence, Tuple

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
    """
    Compute per-class accuracy, precision, recall, and F1.
    Returns dict with metrics and numpy arrays for logging.
    """
    if len(all_preds) == 0 or len(all_targets) == 0:
        return {}, np.zeros(len(classes)), np.zeros(len(classes))
    
    # Convert to numpy if needed
    if hasattr(all_preds, 'cpu'):
        all_preds = all_preds.cpu().numpy()
    if hasattr(all_targets, 'cpu'):
        all_targets = all_targets.cpu().numpy()
    
    # Overall accuracy
    overall_acc = (all_preds == all_targets).mean()
    
    # Per-class metrics
    per_class_acc = np.zeros(len(classes))
    per_class_recall = np.zeros(len(classes))
    per_class_precision = np.zeros(len(classes))
    per_class_f1 = np.zeros(len(classes))
    per_class_support = np.zeros(len(classes))
    
    for i, class_name in enumerate(classes):
        # True positives, false positives, false negatives
        mask_true = (all_targets == i)
        mask_pred = (all_preds == i)
        
        tp = ((all_targets == i) & (all_preds == i)).sum()
        fp = ((all_targets != i) & (all_preds == i)).sum()
        fn = ((all_targets == i) & (all_preds != i)).sum()
        
        # Support (number of true instances)
        support = mask_true.sum()
        per_class_support[i] = support
        
        if support > 0:
            # Accuracy for this class (TP / total true instances)
            per_class_acc[i] = tp / support
            # Recall = TP / (TP + FN)
            per_class_recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Precision = TP / (TP + FP)  
        if (tp + fp) > 0:
            per_class_precision[i] = tp / (tp + fp)
        
        # F1 score
        if per_class_precision[i] + per_class_recall[i] > 0:
            per_class_f1[i] = 2 * (per_class_precision[i] * per_class_recall[i]) / (per_class_precision[i] + per_class_recall[i])
    
    metrics = {
        'overall_acc': overall_acc,
        'per_class_acc': per_class_acc,
        'per_class_precision': per_class_precision, 
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'per_class_support': per_class_support,
        'macro_acc': per_class_acc.mean(),
        'macro_precision': per_class_precision.mean(),
        'macro_recall': per_class_recall.mean(), 
        'macro_f1': per_class_f1.mean()
    }
    
    return metrics, per_class_acc, per_class_support

def gather_predictions_ddp(local_preds, local_targets, world_size, is_main_process):
    """
    Gather predictions and targets from all DDP ranks.
    Returns concatenated tensors on main process, empty tensors on others.
    """
    if not dist.is_initialized():
        return local_preds, local_targets
    
    # Gather predictions and targets from all ranks
    gathered_preds = [torch.zeros_like(local_preds) for _ in range(world_size)]
    gathered_targets = [torch.zeros_like(local_targets) for _ in range(world_size)]
    
    dist.all_gather(gathered_preds, local_preds)
    dist.all_gather(gathered_targets, local_targets)
    
    # Concatenate on main process only
    if is_main_process:
        all_preds = torch.cat(gathered_preds, dim=0)
        all_targets = torch.cat(gathered_targets, dim=0)
    else:
        all_preds = torch.empty(0)
        all_targets = torch.empty(0)
    
    return all_preds, all_targets

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
from typing import Optional, Sequence, Tuple

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
            head_module.bias.copy_(torch.from_numpy(b).to(head_module.bias.device).to(head_module.bias.dtype))