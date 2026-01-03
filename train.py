#!/usr/bin/env python3
import argparse, json, os, math, time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models

# -----------------------------
# Dataset
# -----------------------------
class CellsDataset(Dataset):
    def __init__(self, df, class_to_idx, image_col="img_path", label_col="cell_type", augment=False):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.image_col = image_col
        self.label_col = label_col

        # Common 224x224 pipeline for ResNet/ViT
        base = [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.expand(3, *x.shape[1:]) if x.shape[0] == 1 else x),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ]
        if augment:
            self.tx = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomApply([transforms.RandomRotation(degrees=10)], p=0.25),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.3),
                *base,
                transforms.RandomErasing(p=0.15, scale=(0.02, 0.08), value=0),
            ])
        else:
            self.tx = transforms.Compose([
                transforms.Resize(224, interpolation=Image.BILINEAR),
                transforms.CenterCrop(224),
                *base
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_path = row[self.image_col]
        y = self.class_to_idx[row[self.label_col]]
        img = Image.open(img_path).convert("L")
        x = self.tx(img)
        return x, y

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Better reproducibility; can be a bit slower
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_weighted_sampler(df, label_col):
    counts = df[label_col].value_counts().to_dict()
    weights = df[label_col].map(lambda c: 1.0 / max(1, counts.get(c, 0))).values
    return WeightedRandomSampler(weights=torch.DoubleTensor(weights),
                                 num_samples=len(weights),
                                 replacement=True)

def load_splits(train_csv, val_csv=None, test_csv=None, image_col="img_path", label_col="cell_type",
                class_to_idx=None):
    train_df = pd.read_csv(train_csv)
    vdf = pd.read_csv(val_csv) if val_csv else None
    tdf = pd.read_csv(test_csv) if test_csv else None

    # Filter to allowed classes and existing files
    allowed = set(class_to_idx.keys())
    def _clean(df):
        df = df[df[label_col].isin(allowed)].copy()
        df = df[df[image_col].map(lambda p: Path(str(p)).is_file())].copy()
        return df.reset_index(drop=True)

    train_df = _clean(train_df)
    if vdf is not None: vdf = _clean(vdf)
    if tdf is not None: tdf = _clean(tdf)

    return train_df, vdf, tdf

def build_model(backbone, num_classes):
    backbone = backbone.lower()
    if backbone in ("resnet18", "resnet-18"):
        weights = getattr(models, "ResNet18_Weights", None)
        weights = weights.IMAGENET1K_V1 if weights else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model, 224

    if backbone in ("resnet50", "resnet-50"):
        weights = getattr(models, "ResNet50_Weights", None)
        # V2 slightly better if available
        if weights and hasattr(weights, "IMAGENET1K_V2"):
            w = weights.IMAGENET1K_V2
        else:
            w = weights.IMAGENET1K_V1 if weights else None
        model = models.resnet50(weights=w)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model, 224

    if backbone in ("vit_b16", "vit-b16", "vit"):
        weights = getattr(models, "ViT_B_16_Weights", None)
        weights = weights.IMAGENET1K_V1 if weights else None
        model = models.vit_b_16(weights=weights)
        in_dim = model.heads.head.in_features
        model.heads.head = nn.Linear(in_dim, num_classes)
        return model, 224

    if backbone in ("dinov2_vitb14", "dinov2-b14", "dino", "dinov2"):
        try:
            import timm
            model = timm.create_model("vit_base_patch14_dinov2", pretrained=True, num_classes=num_classes)
            return model, 224
        except Exception as e:
            print(f"[WARN] timm or dinov2 model unavailable ({e}); falling back to ViT-B/16.")
            return build_model("vit_b16", num_classes)

    raise ValueError(f"Unknown backbone: {backbone}")

def accuracy(pred, target):
    return (pred == target).mean()

def compute_confusion(preds, gts, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, g in zip(preds, gts):
        cm[g, p] += 1
    return cm

def macro_f1_from_confusion(cm):
    eps = 1e-9
    K = cm.shape[0]
    f1s = []
    for k in range(K):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        prec = tp / (tp + fp + eps)
        rec  = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        f1s.append(f1)
    return float(np.mean(f1s))

# -----------------------------
# Train / Eval
# -----------------------------
def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    n = 0
    preds_all, gts_all = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                logits = model(x)
                loss = criterion(logits, y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            with torch.inference_mode():
                logits = model(x)
                loss = criterion(logits, y)

        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)

        preds_all.extend(logits.argmax(1).detach().cpu().numpy().tolist())
        gts_all.extend(y.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(1, n)
    acc = accuracy(np.array(preds_all), np.array(gts_all))
    cm = compute_confusion(np.array(preds_all), np.array(gts_all), model.num_classes if hasattr(model, "num_classes") else None)
    return avg_loss, acc, preds_all, gts_all, cm

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Train nuclei classifier on combined splits.")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", default=None)
    ap.add_argument("--class_to_idx", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--image_col", default="img_path", help="img_path or img_path_uniform")
    ap.add_argument("--label_col", default="cell_type")
    ap.add_argument("--backbone", default="vit_b16",
                    choices=["resnet18","resnet50","vit_b16","dinov2_vitb14"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--cosine", action="store_true")
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--freeze_backbone_epochs", type=int, default=0, help="Freeze backbone for first N epochs")
    ap.add_argument("--use_weighted_sampler", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true", help="Use mixed precision")
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.class_to_idx, "r") as f:
        class_to_idx = json.load(f)
    classes = sorted(class_to_idx, key=lambda k: class_to_idx[k])
    num_classes = len(classes)

    # Data
    train_df, val_df, test_df = load_splits(
        args.train_csv, args.val_csv, args.test_csv,
        image_col=args.image_col, label_col=args.label_col,
        class_to_idx=class_to_idx
    )

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {0 if test_df is None else len(test_df)}")
    print("Classes:", classes)

    train_ds = CellsDataset(train_df, class_to_idx, args.image_col, args.label_col, augment=True)
    val_ds   = CellsDataset(val_df,   class_to_idx, args.image_col, args.label_col, augment=False)
    test_ds  = CellsDataset(test_df,  class_to_idx, args.image_col, args.label_col, augment=False) if test_df is not None else None

    if args.use_weighted_sampler:
        sampler = make_weighted_sampler(train_df, args.label_col)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers>0)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers>0)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers>0)
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers>0)

    # Model
    model, input_size = build_model(args.backbone, num_classes)
    model.num_classes = num_classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optionally freeze backbone for warmup
    backbone_params = []
    head_params = []
    # Heuristic: for torchvision ResNet, final layer is 'fc'; for ViT it's model.heads
    for n, p in model.named_parameters():
        if ("fc.weight" in n or "fc.bias" in n or "heads.head" in n or n.endswith(".head.weight") or n.endswith(".head.bias")):
            head_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr}
    ], lr=args.lr, weight_decay=args.weight_decay)

    total_epochs = args.epochs
    warmup = max(0, min(args.warmup_epochs, total_epochs-1)) if args.cosine else 0
    if args.cosine:
        def lr_lambda(epoch):
            if epoch < warmup:
                return float(epoch + 1) / float(max(1, warmup))
            progress = (epoch - warmup) / float(max(1, total_epochs - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    # Class-weighted CE based on train counts
    train_counts = train_df[args.label_col].value_counts().to_dict()
    N = len(train_df); K = num_classes
    cls_weights = torch.tensor(
        [N / (K * max(1, train_counts.get(c, 0))) for c in classes],
        dtype=torch.float, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=cls_weights)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_val_f1 = -1.0
    best_path = out_dir / f"best_{args.backbone}.pt"

    # Optionally freeze backbone for first N epochs
    def set_backbone_requires_grad(flag: bool):
        for p in backbone_params:
            p.requires_grad = flag

    if args.freeze_backbone_epochs > 0:
        set_backbone_requires_grad(False)
        print(f"[Init] Freezing backbone for first {args.freeze_backbone_epochs} epochs.")

    log_file = out_dir / "train_log.txt"
    with open(log_file, "a") as lf:
        lf.write(f"Args: {vars(args)}\n")
        lf.write(f"Classes: {classes}\n")

    for epoch in range(1, total_epochs+1):
        t0 = time.time()
        if args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs + 1:
            set_backbone_requires_grad(True)
            print(f"[Epoch {epoch}] Unfroze backbone.")

        # Train
        tr_loss, tr_acc, _, _, _ = run_epoch(model, train_loader, device, criterion, optimizer, scaler)

        # Val
        with torch.inference_mode():
            va_loss, va_acc, va_preds, va_gts, va_cm = run_epoch(model, val_loader, device, criterion)
            va_f1 = macro_f1_from_confusion(va_cm)

        if scheduler is not None:
            scheduler.step()

        dt = time.time() - t0
        msg = (f"Epoch {epoch:02d}/{total_epochs} | "
               f"train_loss {tr_loss:.4f}  train_acc {tr_acc:.3f} | "
               f"val_loss {va_loss:.4f}  val_acc {va_acc:.3f}  val_macroF1 {va_f1:.3f} | "
               f"{dt:.1f}s")
        print(msg)
        with open(log_file, "a") as lf:
            lf.write(msg + "\n")

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            torch.save({
                "model": model.state_dict(),
                "classes": classes,
                "class_to_idx": class_to_idx,
                "val_macroF1": best_val_f1,
                "backbone": args.backbone,
                "image_col": args.image_col,
                "label_col": args.label_col,
            }, best_path)
            print(f"  ✓ Saved new best to {best_path}")

    # Final test (if provided)
    if test_loader is not None:
        print("\nEvaluating best checkpoint on TEST...")
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.to(device)
        with torch.inference_mode():
            te_loss, te_acc, te_preds, te_gts, te_cm = run_epoch(model, test_loader, device, criterion)
            te_f1 = macro_f1_from_confusion(te_cm)
        print(f"TEST  | loss {te_loss:.4f}  acc {te_acc:.3f}  macroF1 {te_f1:.3f}")

        # Save confusion & predictions
        np.savetxt(out_dir / "confusion_test.csv", te_cm, fmt="%d", delimiter=",")
        test_ids = test_df.index.values
        pred_labels = [classes[i] for i in te_preds]
        true_labels = [classes[i] for i in te_gts]
        pd.DataFrame({
            "row_idx": test_ids,
            "true": true_labels,
            "pred": pred_labels,
            "img": test_df[args.image_col].values,
            "slide_id": test_df.get("slide_id", pd.Series([""]*len(test_df))).values
        }).to_csv(out_dir / "test_predictions.csv", index=False)
        print(f"Saved test confusion matrix and predictions to {out_dir}")

if __name__ == "__main__":
    main()
