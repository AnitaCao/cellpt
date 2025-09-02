#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def stratified_split(df, label_col, train_frac=0.8, val_frac=0.1, seed=42):
    rng = np.random.default_rng(seed)
    test_frac = 1.0 - train_frac - val_frac
    if test_frac < 0:
        raise ValueError("train_frac + val_frac must be ≤ 1.0")

    # operate on POSITIONS (0..len-1)
    split = np.empty(len(df), dtype=object)
    for cls, grp in df.groupby(label_col, sort=False):
        pos = grp.index.to_numpy()  # positions because df was reset_index
        rng.shuffle(pos)
        n = len(pos)

        n_train = int(round(n * train_frac))
        n_val   = int(round(n * val_frac))
        n_train = min(n_train, n)
        n_val   = min(n_val, n - n_train)

        split[pos[:n_train]] = "train"
        split[pos[n_train:n_train+n_val]] = "val"
        split[pos[n_train+n_val:]] = "test"
    return split

def main():
    ap = argparse.ArgumentParser(description="Stratified split using classes in class_to_idx.json")
    ap.add_argument("--meta_csv", required=True, help="combined_meta_original.csv")
    ap.add_argument("--class_to_idx", required=True, help="class_to_idx.json")
    ap.add_argument("--out_dir", default=None, help="Output dir (default: same as meta_csv)")
    ap.add_argument("--label_col", default="cell_type") # or "cell_type_coarse"
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    args = ap.parse_args()

    meta_path = Path(args.meta_csv)
    out_dir = Path(args.out_dir) if args.out_dir else meta_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.class_to_idx, "r") as f:
        allowed = set(json.load(f).keys())

    df = pd.read_csv(meta_path, low_memory=False)
    if args.label_col not in df.columns:
        raise RuntimeError(f"Label column '{args.label_col}' not found.")

    # Filter to allowed classes and RESET INDEX to 0..N-1
    df = df[df[args.label_col].isin(allowed)].copy().reset_index(drop=True)

    print("Class counts after filtering to allowed labels:\n",
          df[args.label_col].value_counts(), "\n")

    # Stratified split
    df["split"] = stratified_split(
        df, args.label_col,
        train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed
    )

    stem = meta_path.stem  # e.g., combined_meta_original
    base = out_dir / f"{stem}_selected"
    with_splits = Path(f"{base}_with_splits.csv")
    df.to_csv(with_splits, index=False)
    print(f"✓ Wrote: {with_splits}  (rows={len(df)})")

    for name in ["train", "val", "test"]:
        part = df[df["split"] == name].copy()
        out_csv = Path(f"{base}_{name}.csv")
        part.to_csv(out_csv, index=False)
        print(f"  - {name:5s}: {len(part):8d}  ->  {out_csv}")

if __name__ == "__main__":
    main()
