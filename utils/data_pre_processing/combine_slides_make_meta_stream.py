#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import numpy as np


# -----------------------------
# Helpers
# -----------------------------
def norm_label(s: str) -> str:
    # trim and collapse internal whitespace, preserve case
    return " ".join(str(s).strip().split())


def choose_meta(slide_dir: Path, prefer_uniform: bool) -> Path:
    # Try common variants in a sensible order
    m_uniform = slide_dir / "meta" / "nucleus_shapes_uniform_3x.csv"
    m_orig    = slide_dir / "meta" / "nucleus_shapes.csv"
    if prefer_uniform and m_uniform.is_file():
        return m_uniform
    if m_orig.is_file():
        return m_orig
    raise FileNotFoundError(
        f"No meta CSV found in {slide_dir} (checked {m_orig.name}, {m_uniform.name})"
    )


def load_one(
    slide_root: Path,
    slide_name: str,
    prefer_uniform: bool,
    include_unknown: bool,
    verify_paths: bool
) -> pd.DataFrame:
    """
    When prefer_uniform=True:
      - keep both img_path and img_path_uniform if they exist
      - drop a row only if both paths are empty or invalid
    When prefer_uniform=False:
      - keep a single canonical img_path column
    """
    sdir = slide_root / slide_name
    meta_path = choose_meta(sdir, prefer_uniform)
    df = pd.read_csv(meta_path, low_memory=False)

    cols = set(df.columns)

    # Build the list of image columns to keep
    keep_img_cols = []
    if prefer_uniform:
        # keep both if present
        if "img_path" in cols:
            keep_img_cols.append("img_path")
        if "img_path_uniform" in cols:
            keep_img_cols.append("img_path_uniform")
        if not keep_img_cols:
            raise RuntimeError(f"{slide_name}: prefer_uniform=True but no img_path or img_path_uniform columns found")
    else:
        # single canonical column
        if "img_path" in cols:
            keep_img_cols = ["img_path"]
        elif "img_path_uniform" in cols:
            keep_img_cols = ["img_path_uniform"]
        else:
            raise RuntimeError(f"{slide_name}: no img_path or img_path_uniform column found")

    needed = set(keep_img_cols) | {"cell_type", "skipped_reason"}
    missing = needed - cols
    if missing:
        raise RuntimeError(f"{slide_name}: missing columns {sorted(list(missing))}")

    # filter by skipped_reason
    df = df[df["skipped_reason"].fillna("") == ""].copy()

    # keep rows with at least one valid image path
    if verify_paths:
        # expensive but accurate check if requested
        def any_valid_file(row):
            ok = False
            for c in keep_img_cols:
                p = str(row[c]).strip()
                if p and Path(p).is_file():
                    ok = True
                    break
            return ok
        df = df[df.apply(any_valid_file, axis=1)].copy()
    else:
        # fast non empty check
        mask = False
        for c in keep_img_cols:
            mask = mask | (df[c].astype(str).str.len() > 0)
        df = df[mask].copy()

    # drop Unknown unless requested
    df["cell_type"] = df["cell_type"].fillna("Unknown")
    if not include_unknown:
        df = df[df["cell_type"] != "Unknown"].copy()

    # normalize labels
    df["cell_type"] = df["cell_type"].map(norm_label)

    # standardize columns for output
    df["slide_id"] = slide_name

    # If not prefer_uniform and we kept only img_path_uniform, rename it to img_path
    if not prefer_uniform and keep_img_cols == ["img_path_uniform"]:
        df = df.rename(columns={"img_path_uniform": "img_path"})
        keep_img_cols = ["img_path"]

    # Build final column list
    base_cols = ["slide_id", "cell_type", "skipped_reason"]
    # put image columns after slide_id for readability
    out_cols = ["slide_id"] + keep_img_cols + ["cell_type", "skipped_reason"]

    return df[out_cols].reset_index(drop=True)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Two pass, low memory combine of slide metas")
    ap.add_argument("--root", required=True, help="Root dir with <SLIDE>/meta/nucleus_shapes*.csv")
    ap.add_argument("--slides", nargs="+", default=[], help="Slide folder names")
    ap.add_argument("--slides_file", default="", help="Optional text file with one slide per line")
    ap.add_argument("--prefer_uniform", action="store_true",
                    help="If set, keep both img_path and img_path_uniform when available")
    ap.add_argument("--include_unknown", action="store_true", help="Keep rows with cell_type == 'Unknown'")
    ap.add_argument("--verify_paths", action="store_true", help="Check that image paths exist on disk")
    ap.add_argument("--min_count", type=int, default=1, help="Drop classes with < min_count after merging")
    ap.add_argument("--out_csv", required=True, help="Output combined CSV")
    ap.add_argument("--class_map_json", default="", help="Output class_to_idx path (default next to out_csv)")
    ap.add_argument("--pivot_csv", default="", help="Optional per slide by class counts CSV")
    ap.add_argument("--project_to_map", default="",
                    help="Optional existing class_to_idx.json to project to and to fix class order")
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
    for s in slides:
        print(" -", s)

    # Pass 1: per slide filtering and counting
    counts_global = Counter()
    per_slide_counts = defaultdict(Counter)
    total_rows = 0

    for s in slides:
        try:
            df_s = load_one(root, s, args.prefer_uniform, args.include_unknown, args.verify_paths)
            vc = df_s["cell_type"].value_counts()
            total_rows += len(df_s)
            counts_global.update(vc.to_dict())
            per_slide_counts[s].update(vc.to_dict())
            print(f"{s}: kept {len(df_s):,} rows")
        except Exception as e:
            print(f"{s}: SKIP pass1 -> {e}")

    if not counts_global:
        raise SystemExit("Nothing counted in pass one.")

    # Choose kept classes
    if args.project_to_map:
        with open(args.project_to_map, "r") as f:
            fixed_map = json.load(f)
        fixed_set = set(fixed_map.keys())
        keep_classes = fixed_set & set(counts_global.keys())
        dropped = sorted(list(fixed_set - keep_classes))
        if dropped:
            print(f"Warning: classes in provided map but absent in data: {dropped[:10]} ...")
    else:
        if args.min_count > 1:
            keep_classes = {c for c, n in counts_global.items() if n >= args.min_count}
        else:
            keep_classes = set(counts_global.keys())

    print(f"Total rows after per slide filtering: {total_rows:,}")
    print(f"Classes found: {len(counts_global)}. Classes kept: {len(keep_classes)}")

    # Optional pivot export
    if args.pivot_csv:
        all_classes = sorted(counts_global.keys())
        rows = []
        for s in slides:
            row = {"slide_id": s}
            for c in all_classes:
                row[c] = int(per_slide_counts[s].get(c, 0))
            rows.append(row)
        pivot_df = pd.DataFrame(rows)
        Path(args.pivot_csv).parent.mkdir(parents=True, exist_ok=True)
        pivot_df.to_csv(args.pivot_csv, index=False)
        print(f"Wrote pivot counts: {args.pivot_csv}")

    # Pass 2: write combined CSV incrementally
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    header_written = False
    written = 0
    for s in slides:
        try:
            df_s = load_one(root, s, args.prefer_uniform, args.include_unknown, args.verify_paths)
        except Exception as e:
            print(f"{s}: SKIP pass2 -> {e}")
            continue
        df_s = df_s[df_s["cell_type"].isin(keep_classes)].copy()
        if df_s.empty:
            continue
        df_s.to_csv(out_path, mode="a", header=not header_written, index=False)
        header_written = True
        written += len(df_s)
        print(f"{s}: wrote {len(df_s):,}")
    print(f"\n✓ Wrote combined meta: {out_path}  (rows={written:,})")

    # Final counts from combined for stats and map
    counts_final = Counter()
    for ch in pd.read_csv(out_path, usecols=["cell_type"], chunksize=200000):
        counts_final.update(ch["cell_type"].value_counts().to_dict())

    # Class map
    if args.project_to_map:
        fixed_map = json.load(open(args.project_to_map))
        classes_sorted = [k for k, _ in sorted(fixed_map.items(), key=lambda kv: kv[1]) if k in counts_final]
    else:
        classes_sorted = sorted(counts_final.keys())

    class_to_idx = {c: i for i, c in enumerate(classes_sorted)}
    cmap_path = args.class_map_json or str(out_path.with_name("class_to_idx.json"))
    with open(cmap_path, "w") as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"✓ Wrote class_to_idx: {cmap_path}")

    # Consistency guard
    seen = set()
    for ch in pd.read_csv(out_path, usecols=["cell_type"], chunksize=200000):
        seen |= set(ch["cell_type"].unique().tolist())
    extra = sorted(list(seen - set(class_to_idx.keys())))
    missing = sorted(list(set(class_to_idx.keys()) - seen))
    if extra or missing:
        raise RuntimeError(f"Label mismatch after write. extra_in_csv={extra[:10]} missing_in_csv={missing[:10]}")

    # Imbalance metrics
    counts_arr = np.array([counts_final[c] for c in classes_sorted], dtype=np.float64)
    if counts_arr.size:
        ratio = counts_arr.max() / max(1.0, counts_arr.min())
        eff = (counts_arr.sum() ** 2) / (np.square(counts_arr).sum() + 1e-9)
        print("\n==== IMBALANCE METRICS ====")
        print(f"Classes: {len(classes_sorted)}   Total: {int(counts_arr.sum()):,}")
        print(f"Max/Min count ratio: {ratio:.1f}x")
        print(f"Effective number of classes (Simpson): {eff:.2f}")


if __name__ == "__main__":
    main()
