# make_folds_and_downsample_hierarchical.py
import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

CSV_IN = "/hpc/group/jilab/rz179/cellpt/combined/withBackground/combined_meta_human_only_20to1.csv"
OUT_DIR = "/hpc/group/jilab/rz179/cellpt/combined/withBackground/folds_out"
SLIDE_COL = "slide_id"
FINE_COL = "cell_type"
COARSE_COL = "cell_type_coarse"
RNG_SEED = 1337

# 4 fixed test slides - never touched until final evaluation
TEST_SLIDES = [
    "Xenium_V1_hLung_cancer",
    "Xenium_V1_hColon_Non_diseased_Base",
    "Xenium_V1_FFPE_Human_Brain_Glioblastoma",
    "Xenium_V1_hLymphNode_nondiseased",
]

# Redistributed 24 slides for balanced validation sets
FOLDS = {
    1: [
        "Xenium_V1_FFPE_Human_Brain_Alzheimers",
        "Xenium_V1_hLiver_cancer",
        "Xenium_V1_hKidney_nondiseased", 
        "Xenium_V1_hSkin_nondiseased_section_1",
        "Xenium_V1_hBone_nondiseased",
        "Xenium_V1_hBoneMarrow_nondiseased",
    ],
    2: [
        "Xenium_V1_FFPE_Human_Brain_Healthy",
        "Xenium_V1_hLiver_nondiseased",
        "Xenium_V1_hKidney_cancer",
        "Xenium_V1_hHeart_nondiseased", 
        "Xenium_V1_hSkin_nondiseased_section_2",
        "Xenium_V1_hPancreas_nondiseased",
    ],
    3: [
        "Xenium_Preview_Human_Non_diseased_Lung",
        "Xenium_V1_hColon_Non_diseased_Add_on",
        "Xenium_V1_hBoneMarrow_acute_lymphoid",
        "Xenium_human_Pancreas_FFPE",
        "Xenium_human_Lung_Cancer_FFPE", 
        "Xenium_human_Lymph_Node_FFPE",
    ],
    4: [
        "Xenium_Preview_Human_Lung_Cancer",
        "Xenium_V1_hColon_Cancer_Base",
        "Xenium_V1_hPancreas_Cancer_Add_on",
        "Xenium_V1_hTonsil_reactive_follicular_hyperplasia",
        "Xenium_V1_hColon_Cancer_Add_on",
        "Xenium_Prime_Ovarian_Cancer_FFPE",
    ],
}

# Hierarchical balancing targets
TARGET_TOTAL = 350_000

# Coarse-level targets (350K total)
COARSE_TARGETS = {
    "Cancer cells": 66500,           # 19.0%
    "Immune cells": 66500,           # 19.0%
    "Stromal cells": 56000,          # 16.0%
    "Epithelial cells": 42000,       # 12.0%
    "Endothelial cells": 28000,      # 8.0%
    "Muscle cells": 28000,           # 8.0%
    "Stem and progenitor cells": 24500, # 7.0%
    "Glial cells": 21000,            # 6.0%
    "Neurons": 17500,                # 5.0%
}

# Fine-level distribution within each coarse class
FINE_WITHIN_COARSE = {
    "Cancer cells": {
        "Colon cancer cells": 0.33,      # 22,000 cells
        "Ovary cancer cells": 0.27,      # 18,000 cells
        "Lung cancer cells": 0.23,       # 15,000 cells
        "Pancreas cancer cells": 0.17,   # 11,500 cells
    },
    "Immune cells": {
        "T cells": 0.27,                 # 18,000 cells
        "NK cells": 0.26,                # 17,000 cells
        "B cells": 0.23,                 # 15,500 cells
        "Myeloid cells": 0.24,           # 16,000 cells
    },
    "Stromal cells": {
        "Fibroblasts": 0.625,            # 35,000 cells
        "Stromal cells": 0.268,          # 15,000 cells
        "Pericytes": 0.107,              # 6,000 cells
    },
    "Epithelial cells": {
        "Epithelial cells": 1.0,         # 42,000 cells
    },
    "Endothelial cells": {
        "Endothelial cells": 1.0,        # 28,000 cells
    },
    "Muscle cells": {
        "Smooth muscle cells": 1.0,      # 28,000 cells
    },
    "Stem and progenitor cells": {
        "Stem and progenitor cells": 1.0, # 24,500 cells
    },
    "Glial cells": {
        "Oligodendrocytes": 0.714,       # 15,000 cells
        "Astrocytes": 0.143,             # 3,000 cells
        "Microglia": 0.143,              # 3,000 cells
    },
    "Neurons": {
        "Neurons": 1.0,                  # 17,500 cells
    },
}

MIN_PER_SLIDE_PER_CLASS = 50
MAX_DOMAIN_SKEW = 0.8

def mark_domain(slide_id: str) -> str:
    sid = slide_id.lower()
    cancerish = ["cancer", "glioblastoma", "melanoma"]
    if any(tok in sid for tok in cancerish):
        return "cancer"
    return "non"

def split_train_val(df, val_slides):
    val = df[df[SLIDE_COL].isin(val_slides)].copy()
    train = df[~df[SLIDE_COL].isin(val_slides + TEST_SLIDES)].copy()
    return train, val

def compute_availability(train_df):
    # counts per class overall and per slide
    counts_cls = train_df[FINE_COL].value_counts().to_dict()
    per_slide_cls = defaultdict(lambda: defaultdict(int))
    for (sid, cls), n in train_df.groupby([SLIDE_COL, FINE_COL]).size().items():
        per_slide_cls[cls][sid] = int(n)
    return counts_cls, per_slide_cls

def allocate_per_class_domain_aware(cls, cap_c, per_slide_dict, train_df):
    """Domain-aware allocation for a single fine class"""
    df_c = train_df[train_df[FINE_COL] == cls]
    sid_counts = per_slide_dict.get(cls, {})
    if not sid_counts:
        return {}
    
    sid_series = pd.Series(sid_counts, name="n")
    dom_series = df_c[[SLIDE_COL]].drop_duplicates().assign(
        dom=lambda x: x[SLIDE_COL].map(mark_domain)
    ).set_index(SLIDE_COL)["dom"]
    
    # Calculate domain totals
    dom_tot = {}
    for sid, n in sid_series.items():
        dom = dom_series.get(sid, "non")
        dom_tot[dom] = dom_tot.get(dom, 0) + n
    
    # Proportional split with soft bound
    total = sum(sid_series)
    cancer_target = int(round(cap_c * (dom_tot.get("cancer", 0) / total))) if total > 0 else 0
    non_target = cap_c - cancer_target
    
    # Enforce soft bound
    max_side = int(round(cap_c * MAX_DOMAIN_SKEW))
    cancer_target = max(min(cancer_target, max_side), cap_c - max_side)
    non_target = cap_c - cancer_target

    out = {}
    for dom_name, dom_cap in [("cancer", cancer_target), ("non", non_target)]:
        sids = [sid for sid, n in sid_series.items() 
                if dom_series.get(sid, "non") == dom_name and n > 0]
        if not sids or dom_cap <= 0:
            continue
            
        sub = sid_series.loc[sids]
        sub_total = int(sub.sum())
        
        if sub_total <= dom_cap:
            # take all
            for sid, n in sub.items():
                out[sid] = out.get(sid, 0) + int(n)
            continue
        
        # Apply minimum per slide
        floor = min(MIN_PER_SLIDE_PER_CLASS, max(0, dom_cap // len(sids)))
        q = {sid: min(int(floor), int(sub[sid])) for sid in sids}
        assigned = sum(q.values())
        remaining = dom_cap - assigned
        
        # Distribute remaining proportionally
        rem_supply = {sid: max(int(sub[sid]) - q[sid], 0) for sid in sids}
        if remaining > 0 and sum(rem_supply.values()) > 0:
            weights = {sid: rem_supply[sid] / sum(rem_supply.values()) for sid in sids}
            provisional = {sid: q[sid] + int(np.floor(weights[sid] * remaining)) for sid in sids}
            
            # Handle residuals
            resid = dom_cap - sum(provisional.values())
            if resid > 0:
                fracs = sorted(
                    [(weights[sid] * remaining - np.floor(weights[sid] * remaining), sid) 
                     for sid in sids],
                    reverse=True
                )
                for _, sid in fracs[:resid]:
                    provisional[sid] += 1
            q = provisional
        
        out.update({sid: out.get(sid, 0) + q[sid] for sid in sids})
    
    return out

def hierarchical_balance(train_df, coarse_targets, fine_within_coarse):
    """Hierarchical balancing with coarse and fine level control"""
    
    # Check availability
    available_coarse = train_df[COARSE_COL].value_counts().to_dict()
    available_fine = train_df[FINE_COL].value_counts().to_dict()
    
    # Compute per-slide availability
    _, per_slide = compute_availability(train_df)
    
    final_quotas = {}
    
    for coarse_class, coarse_target in coarse_targets.items():
        available = available_coarse.get(coarse_class, 0)
        if available == 0:
            continue
            
        # Adjust target if not enough available
        actual_coarse_target = min(coarse_target, available)
        
        fine_distribution = fine_within_coarse[coarse_class]
        
        for fine_class, fine_ratio in fine_distribution.items():
            fine_target = int(actual_coarse_target * fine_ratio)
            fine_available = available_fine.get(fine_class, 0)
            
            if fine_available == 0:
                continue
                
            # Final target for this fine class
            final_target = min(fine_target, fine_available)
            
            if final_target > 0:
                quota = allocate_per_class_domain_aware(
                    fine_class, final_target, per_slide, train_df
                )
                if quota:
                    final_quotas[fine_class] = quota
    
    return final_quotas

def sample_rows(train_df, per_class_quota):
    rng = np.random.default_rng(RNG_SEED)
    picked_idx = []
    
    for cls, sid_quota in per_class_quota.items():
        for sid, take_n in sid_quota.items():
            sub = train_df[(train_df[FINE_COL] == cls) & (train_df[SLIDE_COL] == sid)]
            if take_n >= len(sub):
                picked_idx.extend(sub.index.tolist())
            else:
                picked_idx.extend(list(rng.choice(sub.index.values, size=take_n, replace=False)))
    
    picked = train_df.loc[picked_idx].copy()
    return picked

def get_distribution_dict(df, col_name):
    """Get distribution as dictionary with counts and percentages"""
    total = len(df)
    counts = df[col_name].value_counts().to_dict()
    percentages = {k: (v / total * 100) for k, v in counts.items()}
    return {
        "counts": counts,
        "percentages": percentages,
        "total_cells": total
    }

def print_distribution_report(df, fold_name, dataset_type):
    """Print both fine and coarse class distributions"""
    total_cells = len(df)
    
    print(f"\n{fold_name} {dataset_type} Distribution Report:")
    print(f"Total cells: {total_cells:,}")
    print("="*60)
    
    # Fine class distribution
    fine_counts = df[FINE_COL].value_counts().sort_values(ascending=False)
    print(f"\nTop 15 Fine Classes ({FINE_COL}):")
    print("-" * 50)
    for i, (cell_type, count) in enumerate(fine_counts.head(15).items()):
        percentage = count / total_cells * 100
        print(f"{cell_type:30s}: {count:6,} ({percentage:5.1f}%)")
    
    # Coarse class distribution
    if COARSE_COL in df.columns:
        coarse_counts = df[COARSE_COL].value_counts().sort_values(ascending=False)
        print(f"\nCoarse Classes ({COARSE_COL}):")
        print("-" * 50)
        for cell_type, count in coarse_counts.items():
            percentage = count / total_cells * 100
            print(f"{cell_type:30s}: {count:6,} ({percentage:5.1f}%)")
    
    print("="*60)

def save_fold_metadata(fold_info, out_dir):
    """Save comprehensive fold metadata to JSON files"""
    
    # Create overall metadata
    overall_metadata = {
        "experiment_info": {
            "random_seed": RNG_SEED,
            "target_total_per_fold": TARGET_TOTAL,
            "min_per_slide_per_class": MIN_PER_SLIDE_PER_CLASS,
            "max_domain_skew": MAX_DOMAIN_SKEW,
            "coarse_targets": COARSE_TARGETS,
            "fine_within_coarse": FINE_WITHIN_COARSE,
            "test_slides": TEST_SLIDES,
            "total_folds": len(FOLDS)
        },
        "fold_assignments": FOLDS,
        "fold_summaries": {}
    }
    
    # Add individual fold summaries
    for fold_k, info in fold_info.items():
        overall_metadata["fold_summaries"][f"fold_{fold_k}"] = {
            "validation_slides": info["val_slides"],
            "train_cell_count": info["train_count"],
            "val_cell_count": info["val_count"],
            "train_fine_distribution": info["train_fine_dist"],
            "train_coarse_distribution": info["train_coarse_dist"],
            "val_fine_distribution": info["val_fine_dist"],
            "val_coarse_distribution": info["val_coarse_dist"]
        }
    
    # Save overall metadata
    with open(f"{out_dir}/experiment_metadata.json", "w") as f:
        json.dump(overall_metadata, f, indent=2)
    
    # Save individual fold metadata files
    for fold_k, info in fold_info.items():
        fold_metadata = {
            "fold_number": fold_k,
            "validation_slides": info["val_slides"],
            "training_slides": info["train_slides"],
            "counts": {
                "training_cells": info["train_count"],
                "validation_cells": info["val_count"]
            },
            "distributions": {
                "training": {
                    "fine_classes": info["train_fine_dist"],
                    "coarse_classes": info["train_coarse_dist"]
                },
                "validation": {
                    "fine_classes": info["val_fine_dist"],
                    "coarse_classes": info["val_coarse_dist"]
                }
            },
            "experiment_params": {
                "random_seed": RNG_SEED,
                "target_total": TARGET_TOTAL,
                "coarse_targets": COARSE_TARGETS,
                "fine_within_coarse": FINE_WITHIN_COARSE
            }
        }
        
        with open(f"{out_dir}/fold_{fold_k}_metadata.json", "w") as f:
            json.dump(fold_metadata, f, indent=2)

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_IN)
    
    # Create final test CSV once
    test_df = df[df[SLIDE_COL].isin(TEST_SLIDES)].copy()
    test_df.to_csv(f"{OUT_DIR}/final_test.csv", index=False)
    print_distribution_report(test_df, "Final Test", "Set")
    
    # Save test set metadata
    test_metadata = {
        "test_slides": TEST_SLIDES,
        "test_cell_count": len(test_df),
        "fine_distribution": get_distribution_dict(test_df, FINE_COL),
        "coarse_distribution": get_distribution_dict(test_df, COARSE_COL) if COARSE_COL in test_df.columns else None
    }
    with open(f"{OUT_DIR}/final_test_metadata.json", "w") as f:
        json.dump(test_metadata, f, indent=2)

    # Collect fold information for metadata
    fold_info = {}

    for k, val_slides in FOLDS.items():
        print(f"\n{'='*80}")
        print(f"PROCESSING FOLD {k}")
        print(f"{'='*80}")
        
        train_df, val_df = split_train_val(df, val_slides)
        
        # Get training slides (all CV slides except validation slides)
        all_cv_slides = [slide for slides in FOLDS.values() for slide in slides]
        train_slides = [slide for slide in all_cv_slides if slide not in val_slides]
        
        # Save val as-is and print its distribution
        val_df.to_csv(f"{OUT_DIR}/fold{k}_val.csv", index=False)
        print_distribution_report(val_df, f"Fold {k}", "Validation")

        # Apply hierarchical balancing
        per_class_quota = hierarchical_balance(train_df, COARSE_TARGETS, FINE_WITHIN_COARSE)

        # Sample rows to create the balanced train CSV
        picked = sample_rows(train_df, per_class_quota)

        # Shuffle and save
        picked = picked.sample(frac=1.0, random_state=RNG_SEED).reset_index(drop=True)
        picked.to_csv(f"{OUT_DIR}/fold{k}_train_balanced.csv", index=False)

        # Print comprehensive distribution report
        print_distribution_report(picked, f"Fold {k}", "Training (Hierarchically Balanced)")
        
        # Collect metadata for this fold
        fold_info[k] = {
            "val_slides": val_slides,
            "train_slides": train_slides,
            "train_count": len(picked),
            "val_count": len(val_df),
            "train_fine_dist": get_distribution_dict(picked, FINE_COL),
            "train_coarse_dist": get_distribution_dict(picked, COARSE_COL) if COARSE_COL in picked.columns else None,
            "val_fine_dist": get_distribution_dict(val_df, FINE_COL),
            "val_coarse_dist": get_distribution_dict(val_df, COARSE_COL) if COARSE_COL in val_df.columns else None
        }
    
    # Save all metadata to JSON files
    save_fold_metadata(fold_info, OUT_DIR)
    print(f"\n{'='*80}")
    print("HIERARCHICAL BALANCING COMPLETE")
    print(f"{'='*80}")
    print("Files created:")
    print(f"  - experiment_metadata.json (overall experiment info)")
    print(f"  - final_test_metadata.json (test set info)")
    for k in FOLDS.keys():
        print(f"  - fold_{k}_metadata.json (fold {k} detailed info)")
        print(f"  - fold{k}_train_balanced.csv & fold{k}_val.csv")

if __name__ == "__main__":
    main()