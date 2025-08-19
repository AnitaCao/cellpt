#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

# Reuse the same helpers from your original script
from combine_slides_make_meta import choose_meta, pick_img_col, load_one

def main():
    ap = argparse.ArgumentParser(description="Two pass, low memory combine of slide metas")
    ap.add_argument("--root", required=True)
    ap.add_argument("--slides", nargs="+", required=False, default=[])
    ap.add_argument("--slides_file", default="")
    ap.add_argument("--prefer_uniform", action="store_true")
    ap.add_argument("--include_unknown", action="store_true")
    ap.add_argument("--min_count", type=int, default=1)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--class_map_json", default="")
    ap.add_argument("--pivot_csv", default="", help="Optional path to write per slide by class counts")
    args = ap.parse_args()

    root = Path(args.root)
    slides = list(args.slides)
    if args.slides_file:
        txt = Path(args.slides_file).read_text().splitlines()
        slides.extend([s.strip() for s in txt if s.strip() and not s.strip().startswith("#")])
    slides = [s for s in slides if s]
    if not slides:
        raise SystemExit("No slides provided")

    print("Slides:")
    for s in slides:
        print(" ", s)

    counts_global = Counter()
    per_slide_counts = defaultdict(Counter)

    # Pass one. Load each slide once. Filter. Count.
    total_rows = 0
    for s in slides:
        try:
            df = load_one(root, s, args.prefer_uniform, args.include_unknown)
        except Exception as e:
            print(f"{s}: skip pass1 due to error -> {e}")
            continue
        vc = df["cell_type"].value_counts()
        total_rows += len(df)
        counts_global.update(vc.to_dict())
        per_slide_counts[s].update(vc.to_dict())
        print(f"{s}: pass1 rows kept {len(df):,}")
        del df

    if not counts_global:
        raise SystemExit("Nothing counted in pass one")

    if args.min_count > 1:
        keep_classes = {c for c, n in counts_global.items() if n >= args.min_count}
    else:
        keep_classes = set(counts_global.keys())

    print(f"Total rows after per slide filtering before min_count: {total_rows:,}")
    print(f"Classes found: {len(counts_global)}. Classes kept with min_count {args.min_count}: {len(keep_classes)}")

    # Optional pivot export without holding full table
    if args.pivot_csv:
        all_classes = sorted(counts_global.keys())
        rows = []
        for s in slides:
            row = {"slide_id": s}
            for c in all_classes:
                row[c] = int(per_slide_counts[s].get(c, 0))
            rows.append(row)
        pivot_df = pd.DataFrame(rows)
        pivot_df.to_csv(args.pivot_csv, index=False)
        print(f"Wrote pivot counts to {args.pivot_csv}")

    # Pass two. Write combined CSV incrementally for kept classes.
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header_written = False
    written = 0
    for s in slides:
        try:
            df = load_one(root, s, args.prefer_uniform, args.include_unknown)
        except Exception as e:
            print(f"{s}: skip pass2 due to error -> {e}")
            continue
        df = df[df["cell_type"].isin(keep_classes)].copy()
        if df.empty:
            continue
        df.to_csv(out_path, mode="a", header=not header_written, index=False)
        header_written = True
        written += len(df)
        print(f"{s}: wrote {len(df):,}")
        del df

    print(f"Wrote combined meta: {out_path} rows {written:,}")

    # Class map and imbalance metrics from kept classes
    kept_counts = np.array([counts_global[c] for c in keep_classes], dtype=np.float64)
    classes_sorted = sorted(keep_classes)
    cmap_path = args.class_map_json or str(out_path.with_name("class_to_idx.json"))
    class_to_idx = {c: i for i, c in enumerate(classes_sorted)}
    with open(cmap_path, "w") as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"Wrote class_to_idx: {cmap_path}")

    if kept_counts.size:
        ratio = kept_counts.max() / max(1.0, kept_counts.min())
        eff = (kept_counts.sum() ** 2) / (np.square(kept_counts).sum() + 1e-9)
        print("Imbalance metrics")
        print(f"Classes {len(classes_sorted)} Total {int(kept_counts.sum()):,}")
        print(f"Max over min ratio {ratio:.1f}x")
        print(f"Effective classes Simpson {eff:.2f}")

if __name__ == "__main__":
    main()
