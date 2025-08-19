#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # headless/back-end safe
import matplotlib.pyplot as plt


def analyze_nucleus_images(meta_csv: str, num_samples: int = 100, out_dir: str = ".", prefix: str = "nucleus"):
    """
    Quick sanity-check of nucleus crops.
    - Uses mask (if available) for area; otherwise intensity threshold fallback.
    - Saves: sampled grid, stats plots, distribution histogram, and a CSV of per-image measurements.
    - All output files will be named with the given prefix.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(meta_csv)

    # keep only usable rows
    df = df[df["skipped_reason"].fillna("") == ""].copy()
    if "cell_type" in df.columns:
        df = df[df["cell_type"].fillna("Unknown") != "Unknown"].copy()
    else:
        raise RuntimeError("meta CSV must contain 'cell_type'")

    if len(df) == 0:
        raise RuntimeError("No usable rows after filtering (skipped_reason/Unknown).")

    # Create distribution plot BEFORE sampling
    dist_path = out_dir / f"{prefix}_cell_type_distribution.png"
    create_distribution_plot(df, dist_path)
    
    classes = sorted(df["cell_type"].unique())
    per_class = max(1, num_samples // max(1, len(classes)))

    # sample rows per class (at least 1 if class has any samples)
    sampled_rows = []
    for ct in classes:
        grp = df[df["cell_type"] == ct]
        n = min(per_class, len(grp))
        if n > 0:
            sampled_rows.extend(grp.sample(n, random_state=42).to_dict("records"))

    print(f"Analyzing {len(sampled_rows)} nucleus images from {len(classes)} classes...")

    results = {
        "cell_type": [],
        "image_path": [],
        "mask_path": [],
        "total_pixels": [],
        "non_black_pixels": [],
        "non_black_ratio": [],
        "mean_intensity": [],
        "std_intensity": [],
        "nucleus_area_px": [],
        "nucleus_diameter_px": [],
        "nucleus_area_um2": [],
        "nucleus_diameter_um": [],
    }

    for row in sampled_rows:
        img_path = row["img_path"]
        mask_path = row.get("mask_path", "")
        cell_type = row["cell_type"]

        try:
            # Load grayscale image
            img = Image.open(img_path).convert("L")
            arr = np.array(img, dtype=np.uint8)

            total = int(arr.size)
            non_black = int((arr > 10).sum())  # fallback; may overwrite with mask
            non_black_ratio = non_black / total if total > 0 else 0.0

            mean_int = float(arr.mean())
            std_int = float(arr.std())

            # Prefer exact nucleus area from mask if available
            nucleus_area_px = None
            if mask_path and Path(mask_path).is_file():
                m = np.array(Image.open(mask_path))
                m_bin = (m > 0)
                nucleus_area_px = int(m_bin.sum())
                # redefine "non-black ratio" to be mask occupancy (signal fraction)
                non_black_ratio = nucleus_area_px / total if total > 0 else 0.0
            else:
                # fallback: simple intensity threshold
                m_bin = (arr > 30)
                nucleus_area_px = int(m_bin.sum())

            # Pixel-based diameter approximation
            nucleus_diam_px = 2.0 * np.sqrt(nucleus_area_px / np.pi) if nucleus_area_px > 0 else 0.0

            # Try micron metrics if spacing columns are present
            area_um2 = np.nan
            diam_um = np.nan
            if "um_per_px_x" in row and "um_per_px_y" in row:
                umx = float(row["um_per_px_x"])
                umy = float(row["um_per_px_y"])
                px_area_to_um2 = umx * umy
                area_um2 = nucleus_area_px * px_area_to_um2
                diam_um = 2.0 * np.sqrt(area_um2 / np.pi) if area_um2 > 0 else np.nan

            results["cell_type"].append(cell_type)
            results["image_path"].append(img_path)
            results["mask_path"].append(mask_path)
            results["total_pixels"].append(total)
            results["non_black_pixels"].append(non_black)
            results["non_black_ratio"].append(non_black_ratio)
            results["mean_intensity"].append(mean_int)
            results["std_intensity"].append(std_int)
            results["nucleus_area_px"].append(nucleus_area_px)
            results["nucleus_diameter_px"].append(nucleus_diam_px)
            results["nucleus_area_um2"].append(area_um2)
            results["nucleus_diameter_um"].append(diam_um)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    analysis_df = pd.DataFrame(results)

    # ------------ summaries ------------
    print("\n" + "=" * 60)
    print("NUCLEUS IMAGE ANALYSIS SUMMARY")
    print("=" * 60)

    # Print distribution summary first
    print_distribution_summary(df)

    avg_non_black = float(analysis_df["non_black_ratio"].mean())
    avg_diam_px = float(analysis_df["nucleus_diameter_px"].mean())
    avg_mean_int = float(analysis_df["mean_intensity"].mean())

    print("\nOverall:")
    print(f"  Avg signal fraction (mask or threshold): {avg_non_black:.3f}")
    print(f"  Avg nucleus diameter (px): {avg_diam_px:.1f}")
    print(f"  Avg mean intensity: {avg_mean_int:.1f}")

    print("\nBy Cell Type:")
    for ct in classes:
        sub = analysis_df[analysis_df["cell_type"] == ct]
        if len(sub) == 0:
            continue
        print(f"  {ct}:")
        print(f"    Signal fraction: {sub['non_black_ratio'].mean():.3f} ± {sub['non_black_ratio'].std():.3f}")
        print(f"    Diameter (px):   {sub['nucleus_diameter_px'].mean():.1f} ± {sub['nucleus_diameter_px'].std():.1f}")
        print(f"    Mean intensity:  {sub['mean_intensity'].mean():.1f} ± {sub['mean_intensity'].std():.1f}")

    # crude similarity indicators
    intensity_by_class = analysis_df.groupby("cell_type")["mean_intensity"].mean()
    size_by_class = analysis_df.groupby("cell_type")["nucleus_diameter_px"].mean()
    print("\nVisual Similarity Indicators:")
    print(f"  Intensity variation across class means (std): {float(intensity_by_class.std()):.2f}")
    print(f"  Size variation across class means (std px):  {float(size_by_class.std()):.2f}")

    # ------------ figures ------------
    grid_path = out_dir / f"{prefix}_samples_grid.png"
    stats_path = out_dir / f"{prefix}_statistics.png"
    create_sample_grid(sampled_rows[:24], analysis_df, grid_path)
    create_statistics_plots(analysis_df, stats_path)
    print(f"\nSaved figures:")
    print(f"  {dist_path}")
    print(f"  {grid_path}")
    print(f"  {stats_path}")

    # ------------ save CSV ------------
    csv_out = out_dir / f"{prefix}_analysis_results.csv"
    analysis_df.to_csv(csv_out, index=False)
    print(f"\nDetailed results saved to: {csv_out}")

    # ------------ quick assessment ------------
    print("\n" + "=" * 60)
    print("ASSESSMENT FOR DEEP LEARNING")
    print("=" * 60)

    if avg_non_black < 0.10:
        print("⚠️  Images are mostly background (low signal fraction).")

    if avg_diam_px < 40:
        print("⚠️  Nuclei are small relative to 224×224 canvas.")

    if float(intensity_by_class.std()) < 10:
        print("⚠️  Class-mean intensities are very similar.")

    if float(size_by_class.std()) < 5:
        print("⚠️  Class-mean sizes are very similar.")

    print("\nSuggestions:")
    print("  • Consider cropping tighter (bbox + margin) or reducing input size (e.g., 128).")
    print("  • Try stronger augmentations (contrast/blur) and class-balanced sampling.")
    print("  • Consider multi-channel inputs (if available) or adding morphological features.")

    return analysis_df


def print_distribution_summary(df: pd.DataFrame):
    """Print cell type distribution summary to console."""
    distribution = df["cell_type"].value_counts().sort_values(ascending=False)
    total_samples = len(df)
    
    print("Cell Type Distribution:")
    print(f"  Total samples: {total_samples}")
    print(f"  Number of classes: {len(distribution)}")
    print()
    
    for cell_type, count in distribution.items():
        percentage = (count / total_samples) * 100
        print(f"  {cell_type:20s}: {count:6d} samples ({percentage:5.1f}%)")
    
    # Check for class imbalance
    max_samples = distribution.max()
    min_samples = distribution.min()
    imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
    
    print(f"\n  Class balance metrics:")
    print(f"    Most frequent class: {max_samples} samples")
    print(f"    Least frequent class: {min_samples} samples")
    print(f"    Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    if imbalance_ratio > 10:
        print("  ⚠️  Significant class imbalance detected!")


def create_distribution_plot(df: pd.DataFrame, out_path: Path):
    """Create a histogram showing the distribution of cell types."""
    distribution = df["cell_type"].value_counts().sort_values(ascending=True)
    
    plt.figure(figsize=(12, max(6, len(distribution) * 0.3)))
    
    # Create horizontal bar plot
    bars = plt.barh(range(len(distribution)), distribution.values)
    
    # Color bars with different colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(distribution)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Customize the plot
    plt.yticks(range(len(distribution)), distribution.index)
    plt.xlabel('Number of Samples')
    plt.title('Distribution of Cell Types', fontsize=14, pad=20)
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (cell_type, count) in enumerate(distribution.items()):
        plt.text(count + max(distribution.values()) * 0.01, i, 
                str(count), va='center', ha='left', fontsize=9)
    
    # Add summary statistics as text box
    total_samples = len(df)
    num_classes = len(distribution)
    max_samples = distribution.max()
    min_samples = distribution.min()
    imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
    
    stats_text = f'Total: {total_samples} samples\nClasses: {num_classes}\nImbalance: {imbalance_ratio:.1f}:1'
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_sample_grid(sampled_rows, analysis_df: pd.DataFrame, out_path: Path):
    """Save a grid with one row per cell type, 6 random samples per row."""
    if len(sampled_rows) == 0:
        return
    
    # Group sampled rows by cell type
    rows_by_type = {}
    for row in sampled_rows:
        cell_type = row["cell_type"]
        if cell_type not in rows_by_type:
            rows_by_type[cell_type] = []
        rows_by_type[cell_type].append(row)
    
    # Sort cell types for consistent ordering
    cell_types = sorted(rows_by_type.keys())
    samples_per_row = 6
    
    # Create figure with appropriate dimensions
    num_rows = len(cell_types)
    fig, axes = plt.subplots(num_rows, samples_per_row, 
                            figsize=(samples_per_row * 2.5, num_rows * 2.5))
    
    # Handle case where there's only one row
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Turn off all axes initially
    for i in range(num_rows):
        for j in range(samples_per_row):
            axes[i, j].axis("off")
    
    for row_idx, cell_type in enumerate(cell_types):
        type_samples = rows_by_type[cell_type]
        
        # Take up to 6 samples (or all if fewer than 6)
        samples_to_show = type_samples[:samples_per_row]
        
        # Add row label (cell type name) on the left
        fig.text(0.02, 1 - (row_idx + 0.5) / num_rows, cell_type, 
                rotation=90, va='center', ha='center', 
                fontsize=12, fontweight='bold')
        
        for col_idx, sample_row in enumerate(samples_to_show):
            try:
                img = Image.open(sample_row["img_path"]).convert("L")
                axes[row_idx, col_idx].imshow(img, cmap="gray")
                
                # Add stats annotation
                stats = analysis_df[analysis_df["image_path"] == sample_row["img_path"]]
                if len(stats) > 0:
                    ratio = float(stats["non_black_ratio"].iloc[0])
                    diam = float(stats["nucleus_diameter_px"].iloc[0])
                    axes[row_idx, col_idx].text(
                        2, img.height - 6,
                        f"r:{ratio:.2f} d:{int(round(diam))}",
                        fontsize=7, color="white", va="bottom", ha="left",
                        bbox=dict(facecolor="black", alpha=0.6, pad=1)
                    )
                
                # Add sample number
                axes[row_idx, col_idx].text(
                    2, 8, f"#{col_idx + 1}",
                    fontsize=7, color="white", va="top", ha="left",
                    bbox=dict(facecolor="blue", alpha=0.6, pad=1)
                )
                
            except Exception as e:
                axes[row_idx, col_idx].text(0.5, 0.5, "ERR", 
                                          ha="center", va="center", 
                                          fontsize=10, color="red")
                continue
    
    plt.suptitle("Sample Images by Cell Type (6 samples per row)", 
                fontsize=14, y=0.98)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.02, 
                       hspace=0.1, wspace=0.05)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_statistics_plots(analysis_df: pd.DataFrame, out_path: Path):
    """Save simple boxplots/scatter of key metrics."""
    if len(analysis_df) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Non-black ratio by class
    analysis_df.boxplot(column="non_black_ratio", by="cell_type", ax=axes[0, 0])
    axes[0, 0].set_title("Signal Fraction by Cell Type")
    axes[0, 0].set_xlabel("")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # Diameter by class (px)
    analysis_df.boxplot(column="nucleus_diameter_px", by="cell_type", ax=axes[0, 1])
    axes[0, 1].set_title("Nucleus Diameter (px) by Cell Type")
    axes[0, 1].set_xlabel("")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # Mean intensity by class
    analysis_df.boxplot(column="mean_intensity", by="cell_type", ax=axes[1, 0])
    axes[1, 0].set_title("Mean Intensity by Cell Type")
    axes[1, 0].set_xlabel("")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # Scatter: diameter vs intensity
    for ct, sub in analysis_df.groupby("cell_type"):
        axes[1, 1].scatter(sub["nucleus_diameter_px"], sub["mean_intensity"], label=ct, alpha=0.7, s=10)
    axes[1, 1].set_xlabel("Nucleus Diameter (px)")
    axes[1, 1].set_ylabel("Mean Intensity")
    axes[1, 1].set_title("Diameter vs Intensity")
    axes[1, 1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    # remove automatic suptitle added by pandas.boxplot
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Analyze nucleus crops for DL readiness.")
    p.add_argument("--meta_csv", required=True, help="Path to nucleus_shapes.csv")
    p.add_argument("--num_samples", type=int, default=100, help="Total samples to analyze (split across classes)")
    p.add_argument("--out_dir", default=".", help="Directory to save figures and CSV")
    p.add_argument("--prefix", default="nucleus", help="Prefix for all output files")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze_nucleus_images(args.meta_csv, args.num_samples, args.out_dir, args.prefix)