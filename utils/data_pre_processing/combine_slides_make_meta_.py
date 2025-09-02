#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np

def choose_meta(slide_dir: Path, prefer_uniform: bool) -> Path:
    m_uniform = slide_dir / "meta" / "nucleus_shapes_uniform_3x_with_coarse.csv"
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
        raise RuntimeError("CSV must contain 'img_path_uniform' when prefer_uniform=True.")
    if "img_path" in df.columns:
        return "img_path"
    raise RuntimeError("CSV must contain either 'img_path' or 'img_path_uniform'.")

def load_one(slide_root: Path, slide_name: str, prefer_uniform: bool) -> pd.DataFrame:
    sdir = slide_root / slide_name
    meta = choose_meta(sdir, prefer_uniform)
    df = pd.read_csv(meta, low_memory=False)
    img_col = pick_img_col(df, prefer_uniform)

    needed = {img_col, "cell_type_coarse", "cell_type", "skipped_reason"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"{slide_name}: CSV missing columns: {missing}")

    # filter
    df = df.copy()
    df = df[(df["skipped_reason"].fillna("") == "")]
    df = df[df[img_col].astype(str).map(lambda p: Path(p).is_file())]

    # keep only what we need
    df = df.rename(columns={img_col: "img_path"})
    df["slide_id"] = slide_name
    return df[["slide_id", "img_path", "cell_type_coarse", "cell_type"]].reset_index(drop=True)

def derive_out_paths(out_csv: str, class_map_json: str | None):
    out_csv = Path(out_csv)
    out_dir = out_csv.parent
    stem = out_csv.stem
    # normalize base name
    for suffix in ["_coarse", "_cell_type"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    coarse_csv = out_dir / f"{stem}_coarse.csv"
    cell_csv   = out_dir / f"{stem}_cell_type.csv"

    if class_map_json:
        coarse_json = Path(class_map_json)
    else:
        coarse_json = out_dir / "class_to_idx_coarse.json"
    cell_json = out_dir / "class_to_idx_cell_type.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    return coarse_csv, cell_csv, coarse_json, cell_json

def is_unknown_like(x: str) -> bool:
    v = str(x).strip().lower()
    return (v == "" or v == "na" or v == "none"
            or v.startswith("unknown"))

def write_version(df_all: pd.DataFrame,
                  label_col: str,
                  out_csv: Path,
                  out_json: Path,
                  min_count: int,
                  include_unknown: bool):
    df = df_all.copy()

    # unknown filter per label
    if not include_unknown:
        df = df[~df[label_col].map(is_unknown_like)]

    # drop rare classes per label
    if min_count > 1:
        vc = df[label_col].value_counts()
        keep = vc[vc >= min_count].index
        before = len(df)
        df = df[df[label_col].isin(keep)].copy()
        print(f"[{label_col}] Dropped rare classes < {min_count}: {before:,} -> {len(df):,}")

    # save CSV
    df.to_csv(out_csv, index=False)
    print(f"✓ Wrote {label_col} CSV: {out_csv}  rows={len(df):,}  classes={df[label_col].nunique()}")

    # class map json for this label
    vc = df[label_col].value_counts()
    classes = sorted(vc.index.tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    with open(out_json, "w") as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"✓ Wrote class map for {label_col}: {out_json}")

    # quick stats
    counts = vc.values.astype(np.float64)
    if counts.size > 0:
        ratio = counts.max() / max(1.0, counts.min())
        eff_num = (counts.sum() ** 2) / (np.square(counts).sum() + 1e-9)
        print(f"[{label_col}] Classes={len(vc)} Total={int(counts.sum()):,} MaxMinRatio={ratio:.1f}x EffClasses={eff_num:.2f}")

    # per slide table for this label
    pivot = pd.pivot_table(df, index="slide_id", columns=label_col, values="img_path",
                           aggfunc="count", fill_value=0)
    print(f"\n[{label_col}] PER-SLIDE x CLASS counts")
    print(pivot)
    print()

def main():
    ap = argparse.ArgumentParser(description="Combine multiple slide metas and emit coarse and fine label datasets.")
    ap.add_argument("--root", type=str, default="/hpc/group/jilab/rz179/cellpt/nucleus_data",
                    help="Root dir with <SLIDE>/meta/*.csv")
    ap.add_argument("--slides", nargs="+", default=[], help="Slide folder names")
    ap.add_argument("--slides_file", default="", help="Optional text file, one slide per line")
    ap.add_argument("--prefer_uniform", action="store_true",
                    help="Prefer *_uniform meta and paths if available")
    ap.add_argument("--include_unknown", action="store_true",
                    help="Keep rows with unknown labels")
    ap.add_argument("--min_count", type=int, default=1,
                    help="Drop classes with count < min_count per output")
    ap.add_argument("--out_csv", type=str,
                    default="/hpc/group/jilab/rz179/cellpt/combined/combined_meta_coarse.csv",
                    help="Output path template. Script will write *_coarse.csv and *_cell_type.csv next to it.")
    ap.add_argument("--class_map_json", default="",
                    help="Optional path for coarse class_to_idx JSON. Fine JSON goes next to it as class_to_idx_cell_type.json")
    args = ap.parse_args()

    root = Path(args.root)
    slides = list(args.slides)
    if args.slides_file:
        txt = Path(args.slides_file).read_text().splitlines()
        slides.extend([s.strip() for s in txt if s.strip() and not s.strip().startswith("#")])
    slides = [s for s in slides if s]
    if not slides:
        raise SystemExit("No slides provided. Use --slides or --slides_file.")

    print("Slides:")
    for s in slides:
        print(" -", s)

    dfs = []
    for s in slides:
        try:
            df_s = load_one(root, s, args.prefer_uniform)
            print(f"{s}: kept {len(df_s):,} rows")
            dfs.append(df_s)
        except Exception as e:
            print(f"{s}: SKIP -> {e}")

    if not dfs:
        raise SystemExit("No data loaded from any slide.")

    df_all = pd.concat(dfs, axis=0, ignore_index=True)

    # derive outputs
    coarse_csv, cell_csv, coarse_json, cell_json = derive_out_paths(args.out_csv,
                                                                    args.class_map_json or None)

    # write both versions
    write_version(df_all, "cell_type_coarse", coarse_csv, coarse_json, args.min_count, args.include_unknown)
    write_version(df_all, "cell_type",        cell_csv,   cell_json,   args.min_count, args.include_unknown)

if __name__ == "__main__":
    main()
