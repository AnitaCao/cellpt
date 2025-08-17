#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np

def choose_meta(slide_dir: Path, prefer_uniform: bool) -> Path:
    m_uniform = slide_dir / "meta" / "nucleus_shapes_uniform_3x.csv"
    m_orig    = slide_dir / "meta" / "nucleus_shapes.csv"
    if prefer_uniform and m_uniform.is_file():
        return m_uniform
    if m_orig.is_file():
        return m_orig
    if m_uniform.is_file():
        return m_uniform
    raise FileNotFoundError(f"No meta CSV found in {slide_dir} (checked {m_orig.name} and {m_uniform.name})")

def pick_img_col(df: pd.DataFrame, prefer_uniform: bool) -> str:
    if prefer_uniform:
        if "img_path_uniform" in df.columns:  
            return "img_path_uniform"
        else:
            raise RuntimeError("CSV must contain 'img_path_uniform' column when prefer_uniform=True.")  
    if "img_path" in df.columns:      
        return "img_path"
    else:
        raise RuntimeError("CSV must contain either 'img_path' or 'img_path_uniform' column.")
  

def load_one(slide_root: Path, slide_name: str, prefer_uniform: bool, include_unknown: bool) -> pd.DataFrame:
    sdir = slide_root / slide_name
    meta = choose_meta(sdir, prefer_uniform)
    df = pd.read_csv(meta, low_memory=False)
    img_col = pick_img_col(df, prefer_uniform)
    needed = {img_col, "cell_type", "skipped_reason"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"{slide_name}: CSV missing columns: {missing}")

    # basic filtering
    df = df[(df["skipped_reason"].fillna("")=="")].copy()
    df = df[df[img_col].astype(str).map(lambda p: Path(p).is_file())].copy()
    if not include_unknown:
        df = df[df["cell_type"].fillna("Unknown") != "Unknown"].copy()

    # keep only what we need + slide_id
    df = df.rename(columns={img_col: "img_path"})
    df["slide_id"] = slide_name
    return df[["slide_id", "img_path", "cell_type", "skipped_reason"]].reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser(description="Combine multiple slide metas and summarize class distribution.")
    ap.add_argument("--root", required=True, help="Root dir that has <SLIDE>/meta/nucleus_shapes*.csv")
    ap.add_argument("--slides", nargs="+", required=False, default=[],
                    help="Slide folder names (space-separated).")
    ap.add_argument("--slides_file", default="", help="Optional text file, one slide per line.")
    ap.add_argument("--prefer_uniform", action="store_true",
                    help="Prefer *_uniform.csv and img_path_uniform if available.")
    ap.add_argument("--include_unknown", action="store_true",
                    help="Keep rows with cell_type=='Unknown'.")
    ap.add_argument("--min_count", type=int, default=1,
                    help="Drop classes with < min_count samples AFTER merging.")
    ap.add_argument("--out_csv", required=True, help="Where to write the combined CSV.")
    ap.add_argument("--class_map_json", default="", help="Optional output path for class_to_idx.json (default: alongside out_csv).")
    args = ap.parse_args()

    root = Path(args.root)
    slides = list(args.slides)
    if args.slides_file:
        txt = Path(args.slides_file).read_text().splitlines()
        slides.extend([s.strip() for s in txt if s.strip() and not s.strip().startswith("#")])
    slides = [s for s in slides if s]
    if not slides:
        raise SystemExit("No slides provided. Use --slides or --slides_file.")

    print("Slides to combine:")
    for s in slides: print(" -", s)

    dfs = []
    for s in slides:
        try:
            df_s = load_one(root, s, args.prefer_uniform, args.include_unknown)
            print(f"{s}: kept {len(df_s):,} rows")
            dfs.append(df_s)
        except Exception as e:
            print(f"{s}: SKIP due to error -> {e}")

    if not dfs:
        raise SystemExit("No data loaded from any slide.")

    df = pd.concat(dfs, axis=0, ignore_index=True)

    # Drop rare classes after merging (optional)
    if args.min_count > 1:
        vc = df["cell_type"].value_counts()
        keep = vc[vc >= args.min_count].index
        before = len(df)
        df = df[df["cell_type"].isin(keep)].copy()
        print(f"Dropped rare classes (<{args.min_count}); rows: {before:,} -> {len(df):,}")

    # Save combined
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\n✓ Wrote combined meta: {args.out_csv}  (rows={len(df):,})")

    # Class stats
    print("\n==== OVERALL CLASS COUNTS ====")
    vc = df["cell_type"].value_counts()
    print(vc.to_string())

    # Per-slide table
    print("\n==== PER-SLIDE × CLASS (counts) ====")
    pivot = pd.pivot_table(df, index="slide_id", columns="cell_type", values="img_path",
                           aggfunc="count", fill_value=0)
    print(pivot)

    # Imbalance quick metrics
    counts = vc.values.astype(np.float64)
    if counts.size > 0:
        ratio = counts.max() / max(1.0, counts.min())
        eff_num = (counts.sum() ** 2) / (np.square(counts).sum() + 1e-9)  # Simpson effective classes
        print("\n==== IMBALANCE METRICS ====")
        print(f"Classes: {len(vc)}   Total: {int(counts.sum()):,}")
        print(f"Max/Min count ratio: {ratio:.1f}x")
        print(f"Effective number of classes (Simpson): {eff_num:.2f}")

    # class_to_idx
    classes = sorted(vc.index.tolist())
    class_to_idx = {c:i for i,c in enumerate(classes)}
    cmap_path = args.class_map_json or str(Path(args.out_csv).with_name("class_to_idx.json"))
    with open(cmap_path, "w") as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"\n✓ Wrote class_to_idx: {cmap_path}")

if __name__ == "__main__":
    main()
