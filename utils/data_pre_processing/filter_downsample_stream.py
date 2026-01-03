#!/usr/bin/env python3
import argparse
import json
import random
import tempfile
from pathlib import Path
import pandas as pd

def reservoir_sample_csv(csv_path, k, columns, out_path, write_header, seed):
    random.seed(seed)
    buf, seen = [], 0
    for chunk in pd.read_csv(csv_path, chunksize=200_000):
        for _, row in chunk.iterrows():
            rowd = row.to_dict()
            if seen < k:
                buf.append(rowd)
            else:
                j = random.randint(0, seen)
                if j < k:
                    buf[j] = rowd
            seen += 1
    if buf:
        pd.DataFrame(buf, columns=columns).to_csv(out_path, mode="a", index=False, header=write_header)
        write_header = False
    return seen, write_header

def main():
    parser = argparse.ArgumentParser(description="Stream filter by JSON label map and downsample any class to at most 100000.")
    parser.add_argument("input_csv", nargs="?", default="/hpc/group/jilab/rz179/cellpt/combined/54s/coarse/combined_meta_full_coarse.csv",
                        help="Input CSV path")
    parser.add_argument("output_csv", nargs="?", default="/hpc/group/jilab/rz179/cellpt/combined/54s/semi_balanced/combined_meta_full_downsampled.csv",
                        help="Output CSV path")
    parser.add_argument("label_map_json", nargs="?", default="/hpc/group/jilab/rz179/cellpt/combined/54s/fine/class_to_idx_cell_type.json",
                        help="Path to class_to_idx_cell_type.json")
    parser.add_argument("--cell_type_col", default="cell_type", help="Column name for cell type")
    parser.add_argument("--chunksize", type=int, default=200000, help="Rows per chunk")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    with open(args.label_map_json, "r") as f:
        label_map = json.load(f)

    keep_set = set(label_map.keys())
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header_written = False
    cols_ref = None
    class_counts = {}

    with tempfile.TemporaryDirectory() as tdir:
        tmp_paths = {ct: Path(tdir) / f"class_{i}.csv" for i, ct in enumerate(sorted(keep_set))}
        # Pass 1
        for chunk in pd.read_csv(args.input_csv, chunksize=args.chunksize):
            if args.cell_type_col not in chunk.columns:
                raise ValueError(f"Column {args.cell_type_col} not found in input CSV")
            chunk = chunk[chunk[args.cell_type_col].isin(keep_set)]
            if chunk.empty:
                continue
            chunk = chunk.assign(label_id=chunk[args.cell_type_col].map(label_map))
            if cols_ref is None:
                cols_ref = list(chunk.columns)
            for ct, df_ct in chunk.groupby(args.cell_type_col):
                df_ct = df_ct[cols_ref]
                df_ct.to_csv(tmp_paths[ct], mode="a", index=False, header=not Path(tmp_paths[ct]).exists())
                class_counts[ct] = class_counts.get(ct, 0) + len(df_ct)
        if cols_ref is None:
            raise SystemExit("No rows matched JSON label map.")
        # Pass 2
        for ct in sorted(class_counts):
            total = class_counts[ct]
            k = min(100000, total)
            write_header = not header_written
            seen, header_written = reservoir_sample_csv(
                csv_path=str(tmp_paths[ct]),
                k=k,
                columns=cols_ref,
                out_path=str(out_path),
                write_header=write_header,
                seed=args.seed,
            )
            assert seen == total
            header_written = True

    print("Done.")
    for ct in sorted(class_counts):
        tot = class_counts[ct]
        print(f"{ct}: seen={tot}, kept={min(100000, tot)}")

if __name__ == "__main__":
    main()
