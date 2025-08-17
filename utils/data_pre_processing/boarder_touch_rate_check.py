#!/usr/bin/env python3
import argparse, cv2
import pandas as pd
from pathlib import Path

def touches_border(mask):
    # mask is uint8; treat >0 as foreground
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    h, w = mask.shape[:2]
    return (
        (mask[0, :] > 0).any() or
        (mask[-1, :] > 0).any() or
        (mask[:, 0] > 0).any() or
        (mask[:, -1] > 0).any()
    )

def main():
    ap = argparse.ArgumentParser(description="Compute border-touch rate from meta CSV")
    ap.add_argument("--meta_csv", required=True,
                    help="Path to nucleus_shapes.csv or nucleus_shapes_uniform.csv")
    ap.add_argument("--use_uniform", action="store_true",
                    help="Use mask_path_uniform if present")
    ap.add_argument("--sample", type=int, default=0,
                    help="Randomly sample N rows (0 = all)")
    ap.add_argument("--by_class", action="store_true",
                    help="Also print per-class rates (needs 'cell_type' column)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.meta_csv)
    col = "mask_path_uniform" if args.use_uniform and "mask_path_uniform" in df.columns else "mask_path"
    if col not in df.columns:
        raise SystemExit(f"Column '{col}' not found in {args.meta_csv}")

    # keep rows with an existing mask file
    df = df[df[col].astype(str).str.len() > 0].copy()
    df = df[df[col].map(lambda p: Path(str(p)).is_file())].copy()
    if "skipped_reason" in df.columns:
        df = df[df["skipped_reason"].fillna("") == ""].copy()

    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=args.seed).copy()

    n = len(df)
    touched_flags = []
    for p in df[col].astype(str):
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            touched_flags.append(False)
            continue
        touched_flags.append(touches_border(m))

    touched = sum(touched_flags)
    print(f"Checked {n} masks (column: {col})")
    print(f"Border-touch count: {touched}  |  Rate: {touched/max(1,n):.1%}")

    if args.by_class and "cell_type" in df.columns:
        df = df.copy()
        df["border_touch"] = touched_flags
        rates = df.groupby("cell_type")["border_touch"].mean().sort_values(ascending=False)
        print("\nBy cell type:")
        for k, v in rates.items():
            print(f"  {k}: {v*100:.1f}%")

if __name__ == "__main__":
    main()
