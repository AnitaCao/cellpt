#!/usr/bin/env python3
import os, json, time, argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd

# --- project imports (same modules your trainer uses) ---
from data.dataset import NucleusCSV, PairedMultiFOV
from utils.utils import (
    worker_init_fn, class_counts_from_df,
    compute_per_class_metrics, print_per_class_summary,
    build_allowed_bitmask_from_meta_json, build_taxonomy_from_meta_json,
    compute_coarse_metrics_from_fine, save_coarse_confusions_from_fine,
    save_confusion_matrices, load_priors_from_csv, compute_logit_prior,
)

# Import your model builders & forward helper from the trainer file
# If your training code is in another module, adjust this import.
from trainer.main_trainer_refactor_v3 import build_backbone_and_heads, forward_batch


def parse_args():
    p = argparse.ArgumentParser("CellPT eval")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .pt saved by training")
    # Evaluation CSVs (choose single-view OR 2-view)
    p.add_argument("--eval_csv", type=str, default=None, help="Single-view CSV")
    p.add_argument("--eval_csv_view1", type=str, default=None, help="Multi-view: view-1 (2.5x) CSV")
    p.add_argument("--eval_csv_view2", type=str, default=None, help="Multi-view: view-2 (10x) CSV")
    p.add_argument("--label_col", type=str, default=None, help="Label column name (defaults to ckpt args.label_col or 'cell_type')")
    p.add_argument("--batch_size", type=int, default=None, help="Override batch size (defaults to ckpt args.batch_size)")
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--use_slide_mask", action="store_true", help="Apply slide-aware masking if masks available")
    # Logit adjustment (optional)
    p.add_argument("--apply_la", action="store_true", help="Apply logit adjustment at eval")
    p.add_argument("--tau", type=float, default=None, help="LA temperature (defaults to ckpt args.logit_tau)")
    p.add_argument("--prior_csv", type=str, default=None, help="CSV with priors (defaults to ckpt args.prior_csv)")
    # Outputs
    p.add_argument("--out_dir", type=str, default=None, help="Where to save confusion matrices & metrics.json")
    p.add_argument("--save_confusions", action="store_true", help="Save PNG confusion matrices")
    return p.parse_args()


def build_eval_dataset(args_ns, class_to_idx, mean, std, label_col):
    """
    Build a no-augment eval dataset from provided CSVs (single or paired).
    """
    mv = int(getattr(args_ns, "multi_view", 0)) == 1
    if mv:
        csv_a = getattr(args_ns, "eval_csv_view1", None) or getattr(args_ns, "val_csv_view1", None)
        csv_b = getattr(args_ns, "eval_csv_view2", None) or getattr(args_ns, "val_csv_view2", None)
        assert csv_a and csv_b, "Multi-view eval requires --eval_csv_view1 and --eval_csv_view2 (or they must exist in the ckpt args)"
        ds = PairedMultiFOV(
            csv_a=csv_a, csv_b=csv_b,
            class_to_idx=class_to_idx, size=args_ns.img_size, mean=mean, std=std,
            label_col=label_col, key_col="cell_id", slide_col="slide_id",
            augment=False, aug_mode="none", same_photometric=True
        )
    else:
        csv_single = getattr(args_ns, "eval_csv", None) or getattr(args_ns, "val_csv", None)
        assert csv_single, "Single-view eval requires --eval_csv (or it must exist in the ckpt args)"
        ds = NucleusCSV(
            csv_single, class_to_idx, size=args_ns.img_size, mean=mean, std=std,
            label_col=label_col, augment=False, return_slide=True
        )
    # Filter rows to known classes (defensive)
    before = len(ds.df)
    ds.df = ds.df[ds.df[label_col].isin(class_to_idx.keys())].reset_index(drop=True)
    removed = before - len(ds.df)
    if removed > 0:
        print(f"[filter] eval: removed {removed} rows not present in class_map")
    return ds


def main():
    args = parse_args()
    def safe_load(ckpt_path: str):
        # Prefer the safer path. Works on PyTorch versions that support it.
        try:
            return torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            # Older PyTorch that doesn't know weights_only → use legacy behavior.
            return torch.load(ckpt_path, map_location="cpu")
        except Exception as e:
            # If the file contains non-whitelisted objects that weights_only can’t load,
            # fall back explicitly to the legacy mode (only do this for trusted files).
            print(f"[eval] weights_only=True failed ({e}); falling back to weights_only=False for this trusted file.")
            return torch.load(ckpt_path, map_location="cpu", weights_only=False)

    ckpt = safe_load(args.ckpt)

    # Recover classes & training args from checkpoint
    classes = ckpt["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    ckpt_args = SimpleNamespace(**ckpt["args"])

    # Merge CLI overrides into eval args namespace
    eval_args = ckpt_args
    # Required fields for building model/dataset
    if args.eval_csv is not None: setattr(eval_args, "eval_csv", args.eval_csv)
    if args.eval_csv_view1 is not None: setattr(eval_args, "eval_csv_view1", args.eval_csv_view1)
    if args.eval_csv_view2 is not None: setattr(eval_args, "eval_csv_view2", args.eval_csv_view2)
    if args.batch_size is not None: setattr(eval_args, "batch_size", args.batch_size)
    if not hasattr(eval_args, "img_size"): setattr(eval_args, "img_size", 224)
    label_col = args.label_col or getattr(eval_args, "label_col", "cell_type")

    out_dir = Path(args.out_dir or Path(args.ckpt).with_suffix("").parent / f"eval_{time.strftime('%Y%m%d-%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    # ---- Build model identical to training ----
    model, mean, std, _, _ = build_backbone_and_heads(eval_args, classes, num_classes=len(classes), device=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # ---- Dataset & loader (no augment) ----
    ds = build_eval_dataset(eval_args, class_to_idx, mean, std, label_col)
    print(f"Eval samples: {len(ds)}")
    loader = DataLoader(
        ds, batch_size=int(getattr(eval_args, "batch_size", 64)), shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=worker_init_fn(getattr(eval_args, "seed", 42))
    )

    # ---- Slide mask (optional) ----
    allowed_bitmask = build_allowed_bitmask_from_meta_json(classes, level="fine", subset="subset_all") \
                      if args.use_slide_mask else None

    # ---- Logit adjustment (optional) ----
    adj = None
    if args.apply_la:
        prior_csv = args.prior_csv or getattr(eval_args, "prior_csv", None)
        if prior_csv:
            counts_np = class_counts_from_df(ds.df, label_col=label_col, classes=classes)
            prior_counts_np = load_priors_from_csv(prior_csv, label_col, class_to_idx, classes, counts_np)
            _, log_prior = compute_logit_prior(prior_counts_np, device)
            tau = args.tau if args.tau is not None else float(getattr(eval_args, "logit_tau", 1.0))
            adj = (tau * log_prior).view(1, -1)
            adj = adj - adj.mean()
            print(f"[LA] apply_la=1, tau={tau:.3f}")
        else:
            print("[LA] apply_la=1 requested, but no prior_csv available; skipping LA.")

    # ---- Eval loop ----
    all_preds, all_targets = [], []
    all_gates = [] 
    val_total = 0
    loss_raw_sum = 0.0
    loss_eff_sum = 0.0
    correct_raw = 0
    correct_eff = 0

    use_coarse_aux = bool(getattr(eval_args, "use_coarse_aux", 0))
    coarse_classes, coarse_to_idx, fine_to_coarse_idx, _ = \
        build_taxonomy_from_meta_json(classes, subset="subset_all", device=device)

    with torch.inference_mode(), torch.amp.autocast('cuda', dtype=amp_dtype, enabled=torch.cuda.is_available()):
        for batch in loader:
            if len(batch) == 3:
                x, y, sid = batch
            else:
                x, y = batch
                sid = None
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pre, _, aux = forward_batch(model, x, epoch=0, args=eval_args, amp_dtype=amp_dtype)
            if aux is not None and "g" in aux:
                all_gates.append(aux["g"].detach().cpu())
            logits = aux["fused_logits"] if aux is not None else model._cosine_head(pre)

            # raw metrics
            loss_raw = F.cross_entropy(logits, y)
            loss_raw_sum += float(loss_raw.item()) * y.size(0)
            correct_raw += int((logits.argmax(1) == y).sum().item())

            # effective logits = (optional) LA + (optional) coarse bias + (optional) slide mask
            logits_eff = logits.clone()

            if adj is not None:
                logits_eff = logits_eff - adj.to(dtype=logits_eff.dtype, device=logits_eff.device)

            if use_coarse_aux and hasattr(model, "coarse_head"):
                # Same soft bias used during training eval (eta warmup is irrelevant at eval → use full eta)
                eta = float(getattr(eval_args, "hier_alpha", 0.2))
                if eta > 0.0:
                    eps = 1e-6
                    zc = model.coarse_head(pre)
                    qg = F.softmax(zc, dim=1).clamp_min(eps).log()
                    ftc = fine_to_coarse_idx.to(device)
                    if (ftc < 0).any():
                        ftc = ftc.clone(); ftc[ftc < 0] = 0
                        bonus = qg[:, ftc]
                        bonus[:, (fine_to_coarse_idx < 0)] = 0.0
                    else:
                        bonus = qg[:, ftc]
                    logits_eff = logits_eff + eta * bonus

            if args.use_slide_mask and allowed_bitmask is not None and sid is not None:
                # Apply slide-level allowed class mask
                from utils.utils import apply_slide_mask
                logits_eff = apply_slide_mask(logits_eff, sid, allowed_bitmask)

            loss_eff = F.cross_entropy(logits_eff, y)
            loss_eff_sum += float(loss_eff.item()) * y.size(0)
            pred_eff = logits_eff.argmax(1)
            correct_eff += int((pred_eff == y).sum().item())

            all_preds.append(pred_eff.detach().cpu())
            all_targets.append(y.detach().cpu())
            val_total += int(y.size(0))
            
        if all_gates:
            all_gates = torch.cat(all_gates)
            all_targets_cat = torch.cat(all_targets)
            gate_df = pd.DataFrame(
                all_gates.numpy(),
                columns=[f"view_{i}" for i in range(all_gates.shape[1])]
            )
            gate_df["target"] = all_targets_cat.numpy()
            gate_df["target_class"] = [classes[t] for t in all_targets_cat.numpy()]
            gate_df.to_csv(out_dir / "gate_weights.csv", index=False)
            
            # Per-class gate analysis
            print("\n==== GATE ANALYSIS ====")
            for cls_idx, cls_name in enumerate(classes):
                mask = all_targets_cat == cls_idx
                if mask.any():
                    cls_gates = all_gates[mask].mean(dim=0)
                    print(f"{cls_name:>30s}: {[f'{g:.3f}' for g in cls_gates.tolist()]}")

    if val_total == 0:
        print("No samples to evaluate. Check your CSV paths and label_col.")
        return

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    acc_raw = correct_raw / val_total
    acc_eff = correct_eff / val_total
    loss_raw = loss_raw_sum / val_total
    loss_eff = loss_eff_sum / val_total

    print("\n==== EVAL SUMMARY ====")
    print(f"Samples         : {val_total}")
    print(f"Acc (raw)       : {acc_raw:.4f}    Loss (raw): {loss_raw:.4f}")
    print(f"Acc (effective) : {acc_eff:.4f}    Loss (eff): {loss_eff:.4f}")

    # Per-class & macro-F1
    pc, _, _ = compute_per_class_metrics(all_preds, all_targets, classes)
    macro_f1 = float(pc.get("macro_f1", 0.0))
    print(f"Macro-F1        : {macro_f1:.4f}")
    print_per_class_summary(pc, classes)

    # Coarse metrics
    coarse_metrics, _ = compute_coarse_metrics_from_fine(all_preds, all_targets, fine_to_coarse_idx, coarse_classes)
    print(f"[coarse] macro_recall={coarse_metrics['coarse_macro_recall']:.4f}  "
          f"overall_acc={coarse_metrics['coarse_overall_acc']:.4f}")

    # Save confusion matrices (optional)
    if args.save_confusions:
        cm_paths = save_confusion_matrices(
            all_targets.numpy(), all_preds.numpy(), classes, out_dir,
            epoch="eval", norms=("true", "none", "pred"), print_table_for="true", annotate=True
        )
        print(f"Saved confusions to: {out_dir}")

    # Save a small JSON summary
    summary = {
        "samples": val_total,
        "acc_raw": acc_raw, "loss_raw": loss_raw,
        "acc_eff": acc_eff, "loss_eff": loss_eff,
        "macro_f1": macro_f1,
        "coarse_overall_acc": float(coarse_metrics["coarse_overall_acc"]),
        "coarse_macro_recall": float(coarse_metrics["coarse_macro_recall"]),
        "apply_la": bool(args.apply_la),
        "use_slide_mask": bool(args.use_slide_mask),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(Path(out_dir, "metrics_eval.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary → {Path(out_dir, 'metrics_eval.json')}")
    
    pred_df = pd.DataFrame({
        "prediction": all_preds.numpy(),
        "target": all_targets.numpy(),
        "pred_class": [classes[p] for p in all_preds.numpy()],
        "true_class": [classes[t] for t in all_targets.numpy()],
        "correct": (all_preds == all_targets).numpy(),
    })
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    print(f"Saved predictions → {out_dir / 'predictions.csv'}")

    # Save per-class metrics to JSON
    with open(out_dir / "per_class_metrics.json", "w") as f:
        json.dump(pc, f, indent=2, default=float)
    
    
    print("Eval Done.")
    

if __name__ == "__main__":
    main()
