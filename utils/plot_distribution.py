#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default="/hpc/group/jilab/rz179/cellpt/combined/54s/combined_meta_full_coarse.csv",
        help="Combined CSV with labels"
    )
    ap.add_argument(
        "--coarse_json",
        default="/hpc/group/jilab/rz179/cellpt/combined/54s/class_to_idx_coarse.json",
        help="Path to class_to_idx_coarse.json"
    )
    ap.add_argument(
        "--fine_json",
        default="/hpc/group/jilab/rz179/cellpt/combined/54s/class_to_idx_cell_type.json",
        help="Path to class_to_idx_cell_type.json"
    )
    ap.add_argument(
        "--outdir",
        default="/hpc/group/jilab/rz179/cellpt/plots",
        help="Where to save plots"
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # load data
    df = pd.read_csv(args.csv)
    print("Loaded:", df.shape)

    with open(args.coarse_json) as f:
        coarse_map = json.load(f)
    with open(args.fine_json) as f:
        fine_map = json.load(f)
    print("Coarse classes:", len(coarse_map))
    print("Fine classes:", len(fine_map))

    # coarse distribution
    counts = df["cell_type_coarse"].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(12,6))
    counts.plot(kind="bar")
    plt.title("Distribution of cell_type_coarse")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(outdir / "dist_coarse.png", dpi=200)
    plt.close()

    # fine distribution (top 50 only)
    counts = df["cell_type"].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(14,6))
    counts.head(50).plot(kind="bar")
    plt.title("Top 50 cell_type")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(outdir / "dist_fine_top50.png", dpi=200)
    plt.close()

    # fine distribution (log scale, all classes)
    plt.figure(figsize=(14,6))
    counts.plot(kind="bar", logy=True)
    plt.title("Distribution of all cell_type (log scale)")
    plt.ylabel("Count (log scale)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(outdir / "dist_fine_all_log.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
