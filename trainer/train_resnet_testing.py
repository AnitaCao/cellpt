#!/usr/bin/env python3
import argparse, json, os, random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models

# -----------------------------
# Dataset
# -----------------------------
class NucleusMetaDataset(Dataset):
    def __init__(self, df, class_to_idx, augment=False):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx

        # ResNet wants 3-channel, ImageNet normalization
        t = [
            transforms.ToTensor(),                         # HWC[0..255] -> CHW float[0..1]
            transforms.Lambda(lambda x: x.expand(3, *x.shape[1:]) if x.shape[0]==1 else x),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225]),
        ]
        if augment:
            aug = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply([transforms.RandomRotation(degrees=10)], p=0.25),
            ]
            self.tx = transforms.Compose(aug + t)
        else:
            self.tx = transforms.Compose(t)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        #img = Image.open(row.img_path).convert("L")  # grayscale
        img = Image.open(row.img_path_uniform).convert("L")
        x = self.tx(img)
        y = self.class_to_idx[row.cell_type]
        return x, y

# -----------------------------
# Utilities
# -----------------------------
def stratified_split(df, label_col, val_frac=0.2, seed=42, min_per_class=1):
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls, grp in df.groupby(label_col):
        idx = grp.index.to_numpy()
        if len(idx) < max(2, min_per_class):
            # too few for a split: send all to train
            train_idx.extend(idx.tolist())
            continue
        n_val = max(1, int(round(len(idx) * val_frac)))
        perm = rng.permutation(idx)
        val_idx.extend(perm[:n_val].tolist())
        train_idx.extend(perm[n_val:].tolist())
    return df.loc[train_idx].copy(), df.loc[val_idx].copy()

def make_weighted_sampler(df, label_col):
    counts = df[label_col].value_counts().to_dict()
    weights = df[label_col].map(lambda c: 1.0 / counts[c]).values
    return WeightedRandomSampler(weights=torch.DoubleTensor(weights),
                                 num_samples=len(weights),
                                 replacement=True)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Train a ResNet classifier from a single-slide meta CSV")
    ap.add_argument("--meta_csv", required=True,
                    help="Path to /.../nucleus_data/<SLIDE>/meta/nucleus_shapes.csv")
    ap.add_argument("--out_dir", required=True, help="Where to save model & mapping")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_count", type=int, default=1, help="Drop classes with < min_count samples")
    ap.add_argument("--include_unknown", action="store_true", help="Keep rows with cell_type=='Unknown'")
    ap.add_argument("--use_weighted_sampler", action="store_true", help="Handle imbalance with weighted sampler")
    
    ap.add_argument("--model", default="resnet18",
                    choices=["resnet18","resnet34","resnet50"],
                    help="Backbone")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cosine", action="store_true", help="Use cosine LR schedule with warmup")
    ap.add_argument("--warmup_epochs", type=int, default=3, help="Num warmup epochs for cosine")
    
    
    args = ap.parse_args()
    
    
    #print all args
    print("Arguments:")
    for k,v in vars(args).items():
        print(f"  {k}: {v}") 
        

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

    os.makedirs(args.out_dir, exist_ok=True)

    # Load & filter meta
    df = pd.read_csv(args.meta_csv)
    
    print("cell type distribution: ")
    print(df['cell_type'].value_counts())
    
    #needed_cols = {"img_path", "cell_type", "skipped_reason"}
    needed_cols = {"img_path_uniform", "cell_type", "skipped_reason"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"CSV missing columns: {missing}")

    # Keep only usable rows
    #df = df[(df["skipped_reason"].fillna("")=="") & df["img_path"].map(lambda p: Path(p).is_file())].copy()
    df = df[(df["skipped_reason"].fillna("")=="") & df["img_path_uniform"].map(lambda p: Path(str(p)).is_file())].copy()
    if not args.include_unknown and "cell_type" in df.columns:
        df = df[df["cell_type"].fillna("Unknown") != "Unknown"].copy()
        
    print("Filteted data distribution:")
    print(df['cell_type'].value_counts())

    # Drop rare classes (optional)
    if args.min_count > 1:
        vc = df["cell_type"].value_counts()
        keep = vc[vc >= args.min_count].index
        df = df[df["cell_type"].isin(keep)].copy()

    if len(df) == 0:
        raise RuntimeError("No samples left after filtering.")

    # Class mapping
    classes = sorted(df["cell_type"].unique())
    class_to_idx = {c:i for i,c in enumerate(classes)}
    with open(Path(args.out_dir, "class_to_idx.json"), "w") as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"Classes ({len(classes)}):", classes[:])

    # Train/val split (stratified)
    train_df, val_df = stratified_split(df, "cell_type", val_frac=args.val_frac, seed=args.seed,
                                        min_per_class=max(2, args.min_count))
    print(f"Train: {len(train_df)}  |  Val: {len(val_df)}")

    # Datasets & loaders
    train_ds = NucleusMetaDataset(train_df, class_to_idx, augment=True)
    val_ds   = NucleusMetaDataset(val_df,   class_to_idx, augment=False)

    if args.use_weighted_sampler:
        sampler = make_weighted_sampler(train_df, "cell_type")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    if args.model == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if hasattr(models, "ResNet18_Weights") else None
        model = models.resnet18(weights=weights)
    elif args.model == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if hasattr(models, "ResNet34_Weights") else None
        model = models.resnet34(weights=weights)
    else:  # resnet50
        # V2 has slightly better top-1 than V1; fall back if not available
        if hasattr(models, "ResNet50_Weights") and hasattr(models.ResNet50_Weights, "IMAGENET1K_V2"):
            weights = models.ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if hasattr(models, "ResNet50_Weights") else None
        model = models.resnet50(weights=weights)

    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(device)
    

    # Train setup    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    if args.cosine:
        total_epochs = args.epochs
        warmup = max(0, min(args.warmup_epochs, total_epochs-1))
        def lr_lambda(epoch):
            if epoch < warmup:
                return float(epoch + 1) / float(max(1, warmup))
            # cosine from warmup -> total_epochs
            progress = (epoch - warmup) / float(max(1, total_epochs - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    best_acc = 0.0
    best_path = str(Path(args.out_dir, f"best_{args.model}.pt"))

    for epoch in range(1, args.epochs+1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:02d}/{args.epochs}  |  lr {current_lr:.6g}", flush=True)
        
        # --- train ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x,y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)

        train_loss = running_loss / max(1,total)
        train_acc  = correct / max(1,total)

        # --- val ---
        model.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for x,y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                pred = logits.argmax(1)
                correct += (pred==y).sum().item()
                total += y.size(0)
        val_acc = correct / max(1,total)

        print(f"Epoch {epoch:02d}/{args.epochs}  |  train_loss {train_loss:.4f}  "
              f"train_acc {train_acc:.3f}  val_acc {val_acc:.3f}")
        
        
        with torch.inference_mode():
            all_preds = []
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                preds = logits.argmax(1)
                all_preds.extend(preds.cpu().numpy())
            
            pred_counts = np.bincount(all_preds, minlength=len(classes))
            print("Validation predictions:")
            for i, (cls, count) in enumerate(zip(classes, pred_counts)):
                print(f"  {cls}: {count} ({count/len(all_preds)*100:.1f}%)")
            

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model": model.state_dict(),
                "class_to_idx": class_to_idx,
                "val_acc": best_acc,
            }, best_path)
            print(f"  ✓ Saved new best to {best_path}")
            
        if scheduler is not None:
            scheduler.step()

    print(f"Done. Best val_acc={best_acc:.3f}. Checkpoint: {best_path}")

if __name__ == "__main__":
    main()
