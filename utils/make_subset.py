#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default="/hpc/group/jilab/rz179/cellpt/combined/54s/combined_meta_full_coarse.csv",
        help="Input full combined CSV"
    )
    ap.add_argument(
        "--label_col",
        default="cell_type_coarse",
        help="Column to stratify on (cell_type_coarse or cell_type)"
    )
    ap.add_argument(
        "--frac",
        type=float,
        default=0.05,
        help="Fraction to sample per class (default: 0.05 for 5%)"
    )
    ap.add_argument(
        "--min_per_class",
        type=int,
        default=50,
        help="Minimum number of samples per class (default: 50)"
    )
    ap.add_argument(
        "--out",
        default="/hpc/group/jilab/rz179/cellpt/combined/54s/combined_meta_subset_5pct.csv",
        help="Output CSV for subset"
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    args = ap.parse_args()

    out = Path(args.out)
    summary_out = out.with_name(out.stem + "_summary.csv")
    plot_out = out.with_name(out.stem + "_fraction_kept.png")

    df = pd.read_csv(args.csv)
    print("Loaded:", df.shape)

    summary_records = []
    groups = []

    for c, g in df.groupby(args.label_col):
        original_n = len(g)
        target_n = int(original_n * args.frac)
        n = max(target_n, args.min_per_class)

        if original_n < args.min_per_class:
            print(f"⚠️  Class '{c}' has only {original_n} < {args.min_per_class}. Keeping all samples.")
            sample = g
            applied_n = original_n
            below_floor = True
        else:
            sample = g.sample(n=n, random_state=args.seed)
            applied_n = len(sample)
            below_floor = False

        groups.append(sample)
        summary_records.append({
            "class": c,
            "original_count": original_n,
            "target_count": target_n,
            "applied_count": applied_n,
            "fraction_kept": round(applied_n / original_n, 4),
            "below_floor": below_floor
        })

    df_sub = pd.concat(groups).reset_index(drop=True)
    print("Subset size:", df_sub.shape)

    # write subset
    out.parent.mkdir(parents=True, exist_ok=True)
    df_sub.to_csv(out, index=False)
    print("✓ Wrote subset:", out)

    # write summary
    df_summary = pd.DataFrame(summary_records).sort_values("original_count", ascending=False)
    df_summary.to_csv(summary_out, index=False)
    print("✓ Wrote summary:", summary_out)

    # plot fraction kept
    plt.figure(figsize=(14,6))
    plt.bar(df_summary["class"], df_summary["fraction_kept"])
    plt.xticks(rotation=90)
    plt.ylabel("Fraction kept")
    plt.title(f"Fraction of samples kept per class (target={args.frac*100:.1f}%, min={args.min_per_class})")
    plt.tight_layout()
    plt.savefig(plot_out, dpi=200)
    plt.close()
    print("✓ Wrote plot:", plot_out)

if __name__ == "__main__":
    main()
