#!/usr/bin/env python3
import argparse, json, csv
from pathlib import Path
from collections import Counter
import numpy as np

def count_classes(csv_path, label_col, allowed):
    counts = Counter()
    with open(csv_path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            lab = row.get(label_col)
            if lab in allowed:
                counts[lab] += 1
    return counts

def make_targets(counts, train_frac, val_frac):
    targets = {}
    for cls, n in counts.items():
        n_train = int(round(n * train_frac))
        n_val   = int(round(n * val_frac))
        n_train = min(n_train, n)
        n_val   = min(n_val, n - n_train)
        targets[cls] = {"train": n_train, "val": n_val, "test": n - n_train - n_val}
    return targets

def stream_assign(csv_path, out_dir, stem, label_col, allowed, targets, seed):
    rng = np.random.default_rng(seed)
    with open(csv_path, newline='') as f_in, \
         open(out_dir / f"{stem}_selected_with_splits.csv", "w", newline='') as f_all, \
         open(out_dir / f"{stem}_selected_train.csv", "w", newline='') as f_tr, \
         open(out_dir / f"{stem}_selected_val.csv", "w", newline='') as f_va, \
         open(out_dir / f"{stem}_selected_test.csv", "w", newline='') as f_te:

        r = csv.DictReader(f_in)
        base_fields = r.fieldnames
        w_all = csv.DictWriter(f_all, fieldnames=base_fields + ["split"])
        w_tr  = csv.DictWriter(f_tr,  fieldnames=base_fields + ["split"])
        w_va  = csv.DictWriter(f_va,  fieldnames=base_fields + ["split"])
        w_te  = csv.DictWriter(f_te,  fieldnames=base_fields + ["split"])
        for w in (w_all, w_tr, w_va, w_te):
            w.writeheader()

        remaining = {c: targets[c].copy() for c in targets}

        for row in r:
            lab = row.get(label_col)
            if lab not in allowed:
                continue
            quotas = remaining[lab]
            rem_total = quotas["train"] + quotas["val"] + quotas["test"]
            if rem_total <= 0:
                split = "test"
            else:
                p_train = quotas["train"] / rem_total
                p_val = quotas["val"] / rem_total
                u = rng.random()
                if u < p_train:
                    split = "train"
                elif u < p_train + p_val:
                    split = "val"
                else:
                    split = "test"
            quotas[split] -= 1

            out_row = dict(row)
            out_row["split"] = split
            w_all.writerow(out_row)
            if split == "train":
                w_tr.writerow(out_row)
            elif split == "val":
                w_va.writerow(out_row)
            else:
                w_te.writerow(out_row)

def main():
    ap = argparse.ArgumentParser(description="Streaming stratified split by class mapping")
    ap.add_argument("--meta_csv", required=True)
    ap.add_argument("--class_to_idx", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--label_col", default="cell_type")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    args = ap.parse_args()

    meta_path = Path(args.meta_csv)
    out_dir = Path(args.out_dir) if args.out_dir else meta_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.class_to_idx) as f:
        allowed = set(json.load(f).keys())

    counts = count_classes(meta_path, args.label_col, allowed)
    print("Class counts after filtering:")
    for k, v in counts.items():
        print(f"{k}: {v}")
    targets = make_targets(counts, args.train_frac, args.val_frac)
    stream_assign(meta_path, out_dir, meta_path.stem, args.label_col, allowed, targets, args.seed)
    print("Done.")

if __name__ == "__main__":
    main()
