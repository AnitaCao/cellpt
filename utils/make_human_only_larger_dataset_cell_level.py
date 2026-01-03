#!/usr/bin/env python3
import math, json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# =========================
# Config
# =========================
CSV_IN  = "/hpc/group/jilab/rz179/cellpt/combined/withBackground/combined_meta_human_only_20to1.csv"
OUT_DIR = "./subset_scaled_600k"  # Change this for your output

SLIDE_COL  = "slide_id"
FINE_COL   = "cell_type"
COARSE_COL = "cell_type_coarse"

# Optional: fully exclude slides from all splits
EXCLUDE_SLIDES = []

# Target subset size - scale this up from 350k
TARGET_N = 500000  # Adjust this to your desired size (e.g., 500k, 600k, 700k)

# Target distribution based on your 350k dataset
# These are the exact proportions from your smaller dataset
FINE_TARGET_PCT = {
    "Smooth muscle cells": 0.0857,
    "Fibroblasts": 0.0857,
    "Oligodendrocytes": 0.0658,
    "Colon cancer cells": 0.0638,
    "Ovary cancer cells": 0.0638,
    "T cells": 0.0584,
    "NK cells": 0.0583,
    "Stem and progenitor cells": 0.0565,
    "Endothelial cells": 0.0565,
    "Epithelial cells": 0.0565,
    "B cells": 0.0553,
    "Myeloid cells": 0.0511,
    "Lung cancer cells": 0.0462,
    "Pericytes": 0.0455,
    "Pancreas cancer cells": 0.0412,
    "Stromal cells": 0.0409,
    "Microglia": 0.0208,
    "Astrocytes": 0.0204,
    "Neurons": 0.0147,
    "Liver cancer cells": 0.0132,
}

# Balancing knobs
ALPHA = 0.75             # tempered availability (not used when FINE_TARGET_PCT is set)
PER_FINE_MAX = None      # Remove cap to allow scaling - set to None or a higher value
PER_SLIDE_SHARE_CAP = 0.40  # ≤40% of a fine's quota may come from any single slide

# Per-slide split ratios
TRAIN_R = 0.70
VAL_R   = 0.15
TEST_R  = 0.15

# Balanced-split safeguard
MIN_CLASS_FOR_VALTEST = 3

# Repro
RNG_SEED = 1337
np.random.seed(RNG_SEED)


# =========================
# Helpers
# =========================
def distro(frame, col):
    vc = frame[col].value_counts()
    pct = (vc / max(1, len(frame)) * 100).round(2)
    return pd.DataFrame({"count": vc, "percent": pct})

def print_report(name, frame):
    print(f"\n{name}: {len(frame):,} rows")
    print("Fine class distribution:")
    print(distro(frame, FINE_COL).to_string())
    if COARSE_COL in frame.columns:
        print("\nCoarse distribution:")
        print(distro(frame, COARSE_COL).to_string())

def water_fill_int(target, headroom, total_needed, power=0.7):
    if total_needed <= 0:
        return
    hr = {k: max(0, headroom.get(k, 0)) for k in target.keys()}
    if sum(hr.values()) == 0:
        return
    weights = {k: (v ** power) for k, v in hr.items()}
    Z = sum(weights.values())
    if Z == 0:
        return
    alloc = {k: int(math.floor(total_needed * weights[k] / Z)) for k in target.keys()}
    assigned = sum(alloc.values())
    resid = total_needed - assigned
    if resid > 0:
        fracs = sorted(((total_needed * weights[k] / Z) - alloc[k], k) for k in target.keys())
        for _, k in reversed(fracs[:resid]):
            alloc[k] += 1
    for k, v in alloc.items():
        target[k] += v

def strat_balanced_by_label(df_slide, label_col, ratios, seed, min_for_valtest=0):
    """Balanced per-slide split with integer per-class quotas and optional min-1 safeguard."""
    counts = df_slide[label_col].value_counts().to_dict()

    def int_quota(counts, r):
        raw = {c: counts[c] * r for c in counts}
        base = {c: int(np.floor(raw[c])) for c in counts}
        resid = int(round(sum(raw.values()))) - sum(base.values())
        if resid > 0:
            fracs = sorted(((raw[c] - base[c], c) for c in counts), reverse=True)
            for _, c in fracs[:resid]:
                base[c] += 1
        return base

    q_train = int_quota(counts, ratios[0])
    q_val   = int_quota(counts, ratios[1])
    q_test  = {c: counts[c] - q_train[c] - q_val[c] for c in counts}

    # Safeguard
    if min_for_valtest > 0:
        for c, total_c in counts.items():
            if total_c >= min_for_valtest:
                if q_val[c] == 0:
                    if q_train[c] > 0:
                        q_train[c] -= 1; q_val[c] += 1
                    elif q_test[c] > 0:
                        q_test[c] -= 1; q_val[c] += 1
                if q_test[c] == 0:
                    if q_train[c] > 0:
                        q_train[c] -= 1; q_test[c] += 1
                    elif q_val[c] > 0:
                        q_val[c] -= 1; q_test[c] += 1
                assert q_train[c] + q_val[c] + q_test[c] == total_c

    # Sample per class according to quotas
    parts = {"train": [], "val": [], "test": []}
    for c, grp in df_slide.groupby(label_col, sort=False):
        idx = grp.sample(frac=1.0, random_state=seed).index.to_list()
        t1 = q_train.get(c, 0); t2 = q_val.get(c, 0)
        parts["train"].append(df_slide.loc[idx[:t1]])
        parts["val"].append(df_slide.loc[idx[t1:t1+t2]])
        parts["test"].append(df_slide.loc[idx[t1+t2:]])
    train = pd.concat(parts["train"], ignore_index=True) if parts["train"] else df_slide.iloc[0:0]
    val   = pd.concat(parts["val"],   ignore_index=True) if parts["val"]   else df_slide.iloc[0:0]
    test  = pd.concat(parts["test"],  ignore_index=True) if parts["test"]  else df_slide.iloc[0:0]
    return train, val, test

def strat_natural_by_cell(df_slide, label_col, ratios, seed):
    """Natural per-slide split: random by cell, no class quotas."""
    n = len(df_slide)
    n_train = int(round(ratios[0] * n))
    n_val   = int(round(ratios[1] * n))
    idx = df_slide.sample(frac=1.0, random_state=seed).index
    train = df_slide.loc[idx[:n_train]]
    val   = df_slide.loc[idx[n_train:n_train+n_val]]
    test  = df_slide.loc[idx[n_train+n_val:]]
    return train, val, test

def dist_dict(df, col):
    total = int(len(df))
    counts = df[col].value_counts().sort_index()
    pct = (counts / max(1, total) * 100).round(4)
    # Convert to native Python types for JSON serialization
    counts_dict = {str(k): int(v) for k, v in counts.to_dict().items()}
    pct_dict = {str(k): float(v) for k, v in pct.to_dict().items()}
    return {"total": total, "counts": counts_dict, "percent": pct_dict, "unique": int(counts.shape[0])}

def per_slide_summary(df, slide_col=SLIDE_COL, fine_col=FINE_COL, coarse_col=COARSE_COL):
    by_slide = {}
    for sid, grp in df.groupby(slide_col, sort=False):
        entry = {"total": int(len(grp))}
        entry["fine"] = dist_dict(grp, fine_col)
        if coarse_col in grp.columns:
            entry["coarse"] = dist_dict(grp, coarse_col)
        by_slide[str(sid)] = entry
    return by_slide

def class_coverage(df, slide_col=SLIDE_COL, fine_col=FINE_COL):
    tbl = (
        df.groupby([slide_col, fine_col]).size().rename("n").reset_index()
          .assign(present=lambda x: (x["n"] > 0).astype(int))
          .pivot(index=slide_col, columns=fine_col, values="present").fillna(0).astype(int)
    )
    return tbl

def confusion_template(labels):
    """Return an empty confusion matrix dataframe with given class labels in fixed order."""
    idx = pd.Index(labels, name="true")
    cols = pd.Index(labels, name="pred")
    return pd.DataFrame(0, index=idx, columns=cols)


# =========================
# Load
# =========================
print("Loading data...")
df_all = pd.read_csv(CSV_IN).dropna(subset=[SLIDE_COL, FINE_COL]).reset_index(drop=True)
if EXCLUDE_SLIDES:
    df = df_all[~df_all[SLIDE_COL].isin(EXCLUDE_SLIDES)].copy()
    df_ext = df_all[df_all[SLIDE_COL].isin(EXCLUDE_SLIDES)].copy()
else:
    df = df_all.copy()
    df_ext = pd.DataFrame(columns=df_all.columns)

print(f"Total available cells: {len(df):,}")

# =========================
# Build scaled subset using target percentages
# =========================
avail_fine = df[FINE_COL].value_counts()

print(f"\nTarget size: {TARGET_N:,} cells")
print("Using distribution from 350k dataset\n")

# Calculate target counts for each fine class
fine_quota = {}
for fcls, pct in FINE_TARGET_PCT.items():
    target_count = int(round(TARGET_N * pct))
    avail_count = avail_fine.get(fcls, 0)
    
    # Cap by availability
    if PER_FINE_MAX is not None:
        cap = min(avail_count, PER_FINE_MAX)
    else:
        cap = avail_count
    
    fine_quota[fcls] = min(target_count, cap)
    
    if target_count > cap:
        print(f"Warning: {fcls} requested {target_count:,} but only {cap:,} available")

# Handle any classes in data but not in target percentages
for fcls in avail_fine.index:
    if fcls not in fine_quota:
        print(f"Warning: {fcls} found in data but not in target distribution, skipping")
        fine_quota[fcls] = 0

requested_fine_quota = {str(k): int(v) for k, v in fine_quota.items()}

# Adjust to hit TARGET_N exactly
assigned = sum(fine_quota.values())
print(f"\nInitial assignment: {assigned:,} cells")

if assigned != TARGET_N:
    diff = TARGET_N - assigned
    print(f"Adjusting by {diff:,} cells to reach target...")
    
    if diff > 0:
        # Need to add more cells - water fill
        if PER_FINE_MAX is not None:
            max_per_fine = {f: min(avail_fine.get(f, 0), PER_FINE_MAX) for f in fine_quota.keys()}
        else:
            max_per_fine = {f: avail_fine.get(f, 0) for f in fine_quota.keys()}
        headroom = {f: max(0, max_per_fine[f] - fine_quota[f]) for f in fine_quota.keys()}
        water_fill_int(fine_quota, headroom, diff, power=0.7)
    else:
        # Need to remove cells
        to_cut = -diff
        for f in sorted(fine_quota, key=lambda k: fine_quota[k], reverse=True):
            if to_cut == 0:
                break
            cut = min(to_cut, fine_quota[f])
            fine_quota[f] -= cut
            to_cut -= cut

print(f"Final assignment: {sum(fine_quota.values()):,} cells\n")

# =========================
# Allocate each fine across slides with per-slide share caps and sample
# =========================
print("Sampling cells from slides...")
by_slide_fine = df.groupby([SLIDE_COL, FINE_COL]).size().rename("n").reset_index()
supply = {f: sub.set_index(SLIDE_COL)["n"].to_dict() for f, sub in by_slide_fine.groupby(FINE_COL)}

picked_idx = []
rng = np.random.default_rng(RNG_SEED)

for fcls, q in fine_quota.items():
    if q <= 0: 
        continue
    sup = supply.get(fcls, {})
    if not sup: 
        continue

    slides = list(sup.keys())
    counts = np.array([sup[s] for s in slides], dtype=np.int64)
    total = counts.sum()
    if total == 0:
        continue

    # proportional by slide availability
    prop = counts / total
    alloc = np.floor(prop * q).astype(int)

    # per-slide absolute cap
    abs_cap = int(math.ceil(q * PER_SLIDE_SHARE_CAP))
    alloc = np.minimum(alloc, np.minimum(counts, abs_cap))

    assigned = int(alloc.sum())
    left = q - assigned

    # water-fill under cap
    head = np.minimum(counts, abs_cap) - alloc
    while left > 0 and head.sum() > 0:
        i = int(np.argmax(head))
        alloc[i] += 1
        head[i] -= 1
        left -= 1

    # relax cap if still left
    if left > 0:
        extra = counts - alloc
        while left > 0 and extra.sum() > 0:
            i = int(np.argmax(extra))
            alloc[i] += 1
            extra[i] -= 1
            left -= 1

    # sample rows for this fine from each slide
    for s, take_n in zip(slides, alloc.tolist()):
        if take_n <= 0: 
            continue
        sub_idx = df.index[(df[SLIDE_COL]==s) & (df[FINE_COL]==fcls)].values
        if take_n >= len(sub_idx):
            picked_idx.extend(sub_idx.tolist())
        else:
            picked_idx.extend(rng.choice(sub_idx, size=take_n, replace=False).tolist())

# Exact size
picked_idx = list(dict.fromkeys(picked_idx))
if len(picked_idx) > TARGET_N:
    picked_idx = rng.choice(picked_idx, size=TARGET_N, replace=False).tolist()
elif len(picked_idx) < TARGET_N:
    remaining = df.index.difference(picked_idx).values
    need = TARGET_N - len(picked_idx)
    if len(remaining) >= need:
        picked_idx += rng.choice(remaining, size=need, replace=False).tolist()
    else:
        print(f"Warning: Could only reach {len(picked_idx):,} cells (short by {TARGET_N - len(picked_idx):,})")

subset_df = df.loc[picked_idx].sample(frac=1.0, random_state=RNG_SEED).reset_index(drop=True)

print(f"Sampled {len(subset_df):,} cells")

# =========================
# Per-slide splits: balanced + natural
# =========================
print("Creating train/val/test splits...")
train_bal_parts, val_bal_parts, test_bal_parts = [], [], []
val_nat_parts, test_nat_parts = [], []

for sid, df_s in subset_df.groupby(SLIDE_COL, sort=False):
    # Balanced per-slide 70/15/15 with safeguard
    tr_b, va_b, te_b = strat_balanced_by_label(
        df_s, FINE_COL, (TRAIN_R, VAL_R, TEST_R), RNG_SEED, min_for_valtest=MIN_CLASS_FOR_VALTEST
    )
    train_bal_parts.append(tr_b); val_bal_parts.append(va_b); test_bal_parts.append(te_b)

    # Natural per-slide 15/15 on the eval pool (no class quotas)
    eval_pool = df_s.drop(tr_b.index, errors="ignore")
    _, va_n, te_n = strat_natural_by_cell(
        eval_pool, FINE_COL, (0.0, VAL_R/(VAL_R+TEST_R), TEST_R/(VAL_R+TEST_R)), RNG_SEED
    )
    val_nat_parts.append(va_n); test_nat_parts.append(te_n)

train_df  = pd.concat(train_bal_parts, ignore_index=True)
val_bal   = pd.concat(val_bal_parts,   ignore_index=True)
test_bal  = pd.concat(test_bal_parts,  ignore_index=True)
val_nat   = pd.concat(val_nat_parts,   ignore_index=True)
test_nat  = pd.concat(test_nat_parts,  ignore_index=True)

# Final shuffle
train_df = train_df.sample(frac=1.0, random_state=RNG_SEED).reset_index(drop=True)
val_bal  = val_bal.sample(frac=1.0,  random_state=RNG_SEED).reset_index(drop=True)
test_bal = test_bal.sample(frac=1.0, random_state=RNG_SEED).reset_index(drop=True)
val_nat  = val_nat.sample(frac=1.0,  random_state=RNG_SEED).reset_index(drop=True)
test_nat = test_nat.sample(frac=1.0, random_state=RNG_SEED).reset_index(drop=True)

# =========================
# Save splits
# =========================
print(f"\nSaving to {OUT_DIR}...")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
subset_df.to_csv(f"{OUT_DIR}/subset_all.csv", index=False)
train_df.to_csv(f"{OUT_DIR}/train.csv", index=False)
val_bal.to_csv(f"{OUT_DIR}/val_bal.csv", index=False)
test_bal.to_csv(f"{OUT_DIR}/test_bal.csv", index=False)
val_nat.to_csv(f"{OUT_DIR}/val_nat.csv", index=False)
test_nat.to_csv(f"{OUT_DIR}/test_nat.csv", index=False)

print(f"Wrote files in {OUT_DIR}")
print_report("TRAIN (balanced)", train_df)
print_report("VAL_bal (balanced)", val_bal)
print_report("TEST_bal (balanced)", test_bal)
print_report("VAL_nat (natural)", val_nat)
print_report("TEST_nat (natural)", test_nat)

# =========================
# Metadata
# =========================
print("\nGenerating metadata...")

exp_info = {
    "created_at": datetime.utcnow().isoformat() + "Z",
    "random_seed": RNG_SEED,
    "target_total": int(TARGET_N),
    "fine_target_pct": FINE_TARGET_PCT,
    "alpha_tempering": float(ALPHA),
    "per_fine_max": int(PER_FINE_MAX) if PER_FINE_MAX is not None else None,
    "per_slide_share_cap": float(PER_SLIDE_SHARE_CAP),
    "split_ratios": {"train": TRAIN_R, "val": VAL_R, "test": TEST_R},
    "exclude_slides": list(EXCLUDE_SLIDES),
    "requested_fine_quota": requested_fine_quota,
    "achieved_fine_counts": {str(k): int(v) for k, v in subset_df[FINE_COL].value_counts().to_dict().items()},
}

splits = {
    "subset_all": subset_df,
    "train_bal":  train_df,
    "val_bal":    val_bal,
    "test_bal":   test_bal,
    "val_nat":    val_nat,
    "test_nat":   test_nat,
}

split_meta = {}
for name, frame in splits.items():
    split_meta[name] = {
        "fine":   dist_dict(frame, FINE_COL),
        "coarse": dist_dict(frame, COARSE_COL) if COARSE_COL in frame.columns else None,
        "slides": per_slide_summary(frame),
    }

# Per-slide class coverage matrices
cov_dir = Path(OUT_DIR) / "coverage"
cov_dir.mkdir(parents=True, exist_ok=True)
for name, frame in splits.items():
    cov = class_coverage(frame)
    cov.to_csv(cov_dir / f"{name}_coverage.csv")

# Confusion-matrix templates
labels_fine = sorted(subset_df[FINE_COL].unique().tolist())
cm_bal = confusion_template(labels_fine)
cm_nat = confusion_template(labels_fine)
cm_dir = Path(OUT_DIR) / "confusion_templates"
cm_dir.mkdir(parents=True, exist_ok=True)
cm_bal.to_csv(cm_dir / "confusion_template_balanced.csv")
cm_nat.to_csv(cm_dir / "confusion_template_natural.csv")

# Master metadata JSON
master_meta = {
    "experiment": exp_info,
    "splits": {k: {
        "totals": {
            "rows": int(len(v)),
            "slides": int(v[SLIDE_COL].nunique()),
            "fine_classes": int(v[FINE_COL].nunique()),
            "coarse_classes": int(v[COARSE_COL].nunique()) if COARSE_COL in v.columns else None,
        },
        "distributions": {
            "fine": split_meta[k]["fine"],
            "coarse": split_meta[k]["coarse"],
        },
        "per_slide": split_meta[k]["slides"],
    } for k, v in splits.items()},
}

with open(Path(OUT_DIR) / "metadata.json", "w") as f:
    json.dump(master_meta, f, indent=2)

# Light audit CSVs
for name, frame in splits.items():
    cols = [SLIDE_COL, FINE_COL] + ([COARSE_COL] if COARSE_COL in frame.columns else [])
    frame[cols].to_csv(Path(OUT_DIR) / f"{name}_light.csv", index=False)

print(f"\nSaved metadata to {OUT_DIR}/metadata.json")
print(f"Coverage CSVs in {cov_dir}/")
print(f"Confusion templates in {cm_dir}/")

# =========================
# README
# =========================
readme = f"""# Scaled Dataset (Preserving 350k Distribution)

Generated: {datetime.utcnow().isoformat()}Z

## Overview
This dataset was created by scaling up the 350k distribution to {TARGET_N:,} cells.

## Files
- `subset_all.csv` — full subset ({len(subset_df):,} cells)
- `train.csv` — training set (70%, {len(train_df):,} cells)
- `val_bal.csv`, `test_bal.csv` — balanced evaluation sets (15% each)
- `val_nat.csv`, `test_nat.csv` — natural evaluation sets (15% each)
- `metadata.json` — full metadata
- `coverage/*.csv` — per-slide class coverage
- `confusion_templates/*.csv` — prediction templates

## Distribution Comparison
Target was to maintain the 350k proportions:
{chr(10).join(f"- {cls}: {pct:.2%}" for cls, pct in sorted(FINE_TARGET_PCT.items(), key=lambda x: -x[1]))}

## Parameters
- Target: {TARGET_N:,} cells
- Per-fine max: {PER_FINE_MAX if PER_FINE_MAX else 'None (unlimited)'}
- Per-slide share cap: {PER_SLIDE_SHARE_CAP:.1%}
- Split: {TRAIN_R:.0%}/{VAL_R:.0%}/{TEST_R:.0%}
- Random seed: {RNG_SEED}

## Notes
- Distribution closely matches the 350k template
- Balanced splits ensure equal representation for evaluation
- Natural splits reflect actual slide distributions
"""
(Path(OUT_DIR) / "README.md").write_text(readme)

print(f"README written to {OUT_DIR}/README.md")
print("\n✓ Done!")