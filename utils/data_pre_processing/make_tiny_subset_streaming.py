#!/usr/bin/env python3
# tools/make_tiny_subset_streaming.py
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def choose_img_col(cols, prefer_uniform: bool):
    if prefer_uniform and "img_path_3x" in cols: return "img_path_3x"
    if "img_path" in cols: return "img_path"
    if "img_path.1" in cols: return "img_path_3x"
    raise RuntimeError("No image path column found (need 'img_path' or 'img_path_3x').")


def normalize_label(s: str) -> str:
    # strip, collapse spaces, lowercase
    return " ".join(str(s).strip().split()).lower()

def build_norm_key_map(class_keys):
    # map normalized -> canonical class name from class_map
    return { normalize_label(k): k for k in class_keys }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--class_map", required=True)
    ap.add_argument("--out_train_csv", required=True)
    ap.add_argument("--out_val_csv", required=True)
    ap.add_argument("--per_class_train", type=int, default=20)
    ap.add_argument("--per_class_val", type=int, default=5)
    ap.add_argument("--prefer_uniform", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunksize", type=int, default=20000)
    ap.add_argument("--skip_fs_check", action="store_true", help="Skip Path.is_file checks for speed")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)

    # load class map and header
    with open(args.class_map, "r") as f:
        class_to_idx = json.load(f)
    class_keys = set(class_to_idx.keys())

    header = pd.read_csv(args.in_csv, nrows=0)
    cols = [c.strip() for c in header.columns.tolist()]
    img_col = choose_img_col(cols, args.prefer_uniform)
    label_col = "cell_type"  # your file has this
    need_cols = [img_col, label_col] + (["skipped_reason"] if "skipped_reason" in cols else [])
    need_set = set(need_cols)

    # normalized key map from class_map
    norm2canon = build_norm_key_map(class_keys)
    want_norm = set(norm2canon.keys())

    # targets
    per_train = int(args.per_class_train)
    per_val   = int(args.per_class_val)
    target_per_class = per_train + per_val

    # buffers
    buf = {k: [] for k in class_keys}  # store rows as dicts

    got_total = 0
    required_total = target_per_class * len(class_keys)

    for chunk in pd.read_csv(args.in_csv, chunksize=args.chunksize):
        chunk.columns = [c.strip() for c in chunk.columns]
        if not need_set.issubset(set(chunk.columns)): 
            continue
        chunk = chunk[list(need_cols)]
       # normalize labels and map to canonical names
        c = chunk.copy()
        c["_norm"] = c[label_col].apply(normalize_label)
        c = c[c["_norm"].isin(want_norm)]
        c["_canon"] = c["_norm"].map(norm2canon)
        # drop skipped if present
        if "skipped_reason" in c.columns:
            c = c[c["skipped_reason"].fillna("") == ""]
        # fast exit if nothing left to collect
        remaining_classes = [k for k in class_keys if len(buf[k]) < target_per_class]
        if not remaining_classes:
            break
        c = c[c["_canon"].isin(remaining_classes)]

        # optional file existence check, but only for rows we might keep
        if not args.skip_fs_check:
            c = c[c[img_col].map(lambda p: Path(str(p)).is_file())]

        # shuffle chunk rows lightly to avoid bias
        c = c.sample(frac=1.0, random_state=rng)

        # take only what each class still needs
        for cls, g in c.groupby("_canon", sort=False):
            need = target_per_class - len(buf[cls])
            if need <= 0: 
                continue
            take = g.iloc[:need]
            # store minimal fields; preserve any extra if you want
            for _, row in take.iterrows():
                buf[cls].append({img_col: row[img_col], "label_col": cls})
                got_total += 1

        if got_total >= required_total:
            break

    # build tiny train and val dataframes
    rows_train, rows_val = [], []
    for cls in sorted(class_keys, key=lambda k: class_to_idx[k]):
        lst = buf[cls]
        if not lst:
            continue
        # split per class
        t = min(per_train, len(lst))
        v = min(per_val, max(0, len(lst) - t))
        rows_train.extend(lst[:t])
        rows_val.extend(lst[t:t+v])

    tiny_train = pd.DataFrame(rows_train)
    tiny_val   = pd.DataFrame(rows_val)

    # sanity print
    def summarize(df, name):
        print(f"\n{name} size: {len(df)}")
        if len(df):
            vc = df["label_col"].value_counts().sort_index()
            for k, v in vc.items():
                print(f"  {k:>28}: {int(v)}")

    summarize(tiny_train, "Tiny TRAIN")
    summarize(tiny_val,   "Tiny VAL")

    # show which classes we missed entirely
    missed = [k for k in class_keys if k not in set(tiny_train.get(label_col, pd.Series())).union(set(tiny_val.get(label_col, pd.Series())))]
    if missed:
        print("\nClasses with zero samples in tiny set:", missed)

    # write
    Path(args.out_train_csv).parent.mkdir(parents=True, exist_ok=True)
    tiny_train.to_csv(args.out_train_csv, index=False)
    tiny_val.to_csv(args.out_val_csv, index=False)
    print(f"\nWrote:\n  {args.out_train_csv}\n  {args.out_val_csv}")

if __name__ == "__main__":
    main()
