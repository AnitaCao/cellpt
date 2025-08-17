# utils.py
from pathlib import Path
from datetime import datetime
import csv
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

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
# CSV metrics logger
# -----------------------------
def init_metrics_logger(out_dir, filename: str = "metrics.csv"):
    path = Path(out_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow([
            "run_id","epoch","train_loss","train_acc",
            "val_loss","val_acc","lr_head","lr_lora",
            "secs","timestamp_utc"
        ])
        f.flush()
    return f, w

def log_metrics(writer, run_id, epoch, train_loss, train_acc,
                val_loss, val_acc, lr_head, lr_lora, secs):
    writer.writerow([
        run_id, epoch,
        f"{train_loss:.6f}", f"{train_acc:.6f}",
        f"{val_loss:.6f}", f"{val_acc:.6f}",
        f"{lr_head:.8f}", f"{lr_lora:.8f}",
        f"{secs:.3f}", datetime.utcnow().isoformat()
    ])
