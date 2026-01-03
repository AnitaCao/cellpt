#!/usr/bin/env python3
"""
eval_unified.py - Unified evaluation script for CellPT/MorphPT

Handles:
  - Multi-view models (with gate analysis)
  - Single-view baselines: ResNet-50, ConvNeXt, DINOv2 (with/without LoRA), Swin

Features:
  - Auto-detects model type from checkpoint
  - Preserves all original CSV columns including image paths
  - Saves predictions with confidence and correctness
  - Gate analysis for multi-view models
  - Confusion matrices
  - Coarse-grained metrics

Usage:
    # Multi-view model
    python eval_unified.py \
        --ckpt /path/to/mv_checkpoint.pt \
        --eval_csv_view1 /path/to/2p5x.csv \
        --eval_csv_view2 /path/to/10x.csv \
        --out_dir /path/to/output \
        --save_confusions

    # Single-view baseline
    python eval_unified.py \
        --ckpt /path/to/sv_checkpoint.pt \
        --eval_csv /path/to/test.csv \
        --out_dir /path/to/output \
        --save_confusions
"""
import os
import json
import time
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --- project imports ---
from data.dataset import NucleusCSV, PairedMultiFOV
from utils.utils import (
    worker_init_fn,
    class_counts_from_df,
    compute_per_class_metrics,
    print_per_class_summary,
    build_allowed_bitmask_from_meta_json,
    build_taxonomy_from_meta_json,
    compute_coarse_metrics_from_fine,
    save_coarse_confusions_from_fine,
    save_confusion_matrices,
    load_priors_from_csv,
    compute_logit_prior,
    apply_slide_mask,
)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_serializable(v) for v in obj]
    return obj


def safe_load_checkpoint(ckpt_path: str):
    """Load checkpoint with fallback for different PyTorch versions."""
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"[eval] weights_only=True failed ({e}); falling back to weights_only=False")
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)


# =============================================================================
# SINGLE-VIEW MODEL BUILDING (for baselines)
# =============================================================================

class CosineClassifier(nn.Module):
    """Cosine similarity classifier with learnable temperature."""
    def __init__(self, in_features, num_classes, init_temp=0.07):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self._s_unconstrained = nn.Parameter(torch.tensor(1.0 / init_temp))
    
    @property
    def scale(self):
        return self._s_unconstrained
    
    def forward(self, x):
        x_norm = F.normalize(x, dim=-1)
        w_norm = F.normalize(self.weight, dim=-1)
        return self._s_unconstrained * (x_norm @ w_norm.T)


class LoRALinear(nn.Module):
    """Linear layer with LoRA adapters that matches checkpoint naming."""
    def __init__(self, in_features, out_features, rank=8, alpha=None, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        if alpha is None:
            alpha = 2.0 * rank
        self.scale = alpha / rank
    
    def forward(self, x):
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = self.lora_up(self.lora_down(x)) * self.scale
        return base_out + lora_out


def get_lora_target_modules_from_state_dict(state_dict):
    """Extract which modules have LoRA adapters from checkpoint keys."""
    lora_modules = set()
    for key in state_dict.keys():
        if ".lora_down.weight" in key:
            base_path = key.replace(".lora_down.weight", "")
            lora_modules.add(base_path)
    return list(lora_modules)


def inject_lora_from_checkpoint(model, state_dict, default_rank=8, alpha=None):
    """Inject LoRA adapters into model at exactly the locations specified in checkpoint."""
    lora_modules = get_lora_target_modules_from_state_dict(state_dict)
    
    if not lora_modules:
        print("[lora] No LoRA modules found in checkpoint")
        return model
    
    print(f"[lora] Found {len(lora_modules)} LoRA modules in checkpoint")
    
    rank = default_rank
    for key in state_dict.keys():
        if ".lora_down.weight" in key:
            rank = state_dict[key].shape[0]
            break
    
    if alpha is None:
        alpha = 2.0 * rank
    scale = alpha / rank
    print(f"[lora] Inferred rank={rank}, alpha={alpha}, scale={scale:.2f}")
    
    for module_path in lora_modules:
        parts = module_path.split(".")
        
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        
        attr_name = parts[-1]
        try:
            old_module = getattr(parent, attr_name)
        except AttributeError:
            print(f"[lora] Warning: Could not find {module_path}")
            continue
        
        if not isinstance(old_module, nn.Linear):
            print(f"[lora] Warning: {module_path} is not nn.Linear, skipping")
            continue
        
        new_module = LoRALinear(
            in_features=old_module.in_features,
            out_features=old_module.out_features,
            rank=rank,
            alpha=alpha,
            bias=old_module.bias is not None,
        )
        
        new_module.weight.data.copy_(old_module.weight.data)
        if old_module.bias is not None:
            new_module.bias.data.copy_(old_module.bias.data)
        
        setattr(parent, attr_name, new_module)
    
    print(f"[lora] Injected LoRA into {len(lora_modules)} modules")
    return model


def infer_backbone_from_state_dict(state_dict):
    """Infer the backbone architecture from checkpoint state_dict keys."""
    keys = list(state_dict.keys())
    
    if any("cls_token" in k for k in keys) or any("patch_embed" in k for k in keys):
        has_lora = any(".lora_down." in k or ".lora_up." in k for k in keys)
        
        if "patch_embed.proj.weight" in keys:
            embed_dim = state_dict["patch_embed.proj.weight"].shape[0]
        elif any("_cosine_head.weight" in k for k in keys):
            embed_dim = state_dict["_cosine_head.weight"].shape[1]
        else:
            embed_dim = None
        
        if embed_dim == 384:
            return "vit_small_patch14_dinov2.lvd142m", has_lora
        elif embed_dim == 768:
            return "vit_base_patch14_dinov2.lvd142m", has_lora
        elif embed_dim == 1024:
            return "vit_large_patch14_dinov2.lvd142m", has_lora
        else:
            return "vit_small_patch14_dinov2.lvd142m", has_lora
    
    if any("layers.0.blocks" in k for k in keys):
        return "swin_base_patch4_window7_224", False
    
    if any("stages" in k for k in keys) or any("downsample_layers" in k for k in keys):
        return "convnext_base", False
    
    if any("layer1" in k for k in keys) and any("conv1" in k for k in keys):
        return "resnet50", False
    
    return None, False


def infer_image_size_from_state_dict(state_dict):
    """Infer the training image size from pos_embed shape."""
    if "pos_embed" not in state_dict:
        return 224
    
    pos_embed_shape = state_dict["pos_embed"].shape
    n_tokens = pos_embed_shape[1]
    n_patches = n_tokens - 1
    
    patches_per_side = int(round(n_patches ** 0.5))
    if patches_per_side * patches_per_side == n_patches:
        img_size = patches_per_side * 14
        print(f"[info] Inferred img_size={img_size} from pos_embed shape {pos_embed_shape}")
        return img_size
    
    return 224


def build_sv_model(backbone_name, num_classes, use_cosine=True, has_lora=False, 
                   lora_rank=8, lora_alpha=None, img_size=224, state_dict=None):
    """Build single-view model with proper architecture."""
    import timm
    
    if lora_alpha is None:
        lora_alpha = 2.0 * lora_rank
    
    print(f"[model] Building {backbone_name}, cosine={use_cosine}, lora={has_lora}, img_size={img_size}")
    if has_lora:
        print(f"[model] LoRA config: rank={lora_rank}, alpha={lora_alpha}, scale={lora_alpha/lora_rank:.2f}")
    
    if "dinov2" in backbone_name.lower() or "vit" in backbone_name.lower():
        if "small" in backbone_name.lower():
            timm_name = "vit_small_patch14_dinov2.lvd142m"
            embed_dim = 384
        elif "base" in backbone_name.lower():
            timm_name = "vit_base_patch14_dinov2.lvd142m"
            embed_dim = 768
        elif "large" in backbone_name.lower():
            timm_name = "vit_large_patch14_dinov2.lvd142m"
            embed_dim = 1024
        else:
            timm_name = backbone_name
            embed_dim = 384
        
        model = timm.create_model(timm_name, pretrained=False, num_classes=0, img_size=img_size)
        
        if has_lora and state_dict is not None:
            model = inject_lora_from_checkpoint(model, state_dict, default_rank=lora_rank, alpha=lora_alpha)
        
        if use_cosine:
            model._cosine_head = CosineClassifier(embed_dim, num_classes)
            model.head = model._cosine_head
        else:
            model.head = nn.Linear(embed_dim, num_classes)
        
        def new_forward(x):
            features = model.forward_features(x)
            if features.dim() == 3:
                features = features[:, 0]
            if hasattr(model, '_cosine_head'):
                return model._cosine_head(features)
            return model.head(features)
        model.forward = new_forward
        
    elif "convnext" in backbone_name.lower():
        model = timm.create_model(backbone_name, pretrained=False, num_classes=num_classes)
        if use_cosine:
            in_features = model.head.fc.in_features if hasattr(model.head, 'fc') else model.head.in_features
            if hasattr(model.head, 'fc'):
                model.head.fc = CosineClassifier(in_features, num_classes)
            else:
                model.head = CosineClassifier(in_features, num_classes)
        
    elif "swin" in backbone_name.lower():
        model = timm.create_model(backbone_name, pretrained=False, num_classes=num_classes)
        if use_cosine:
            in_features = model.head.fc.in_features if hasattr(model.head, 'fc') else model.head.in_features
            if hasattr(model.head, 'fc'):
                model.head.fc = CosineClassifier(in_features, num_classes)
            else:
                model.head = CosineClassifier(in_features, num_classes)
        
    else:  # ResNet and others
        model = timm.create_model(backbone_name, pretrained=False, num_classes=num_classes)
        if use_cosine and hasattr(model, 'fc'):
            in_features = model.fc.in_features
            model.fc = CosineClassifier(in_features, num_classes)
    
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    return model, mean, std


def load_weights_flexible(model, state_dict):
    """Load weights with flexible key matching."""
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    
    common_keys = model_keys & ckpt_keys
    missing_in_ckpt = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys
    
    print(f"[load] Common keys: {len(common_keys)}, Missing: {len(missing_in_ckpt)}, Unexpected: {len(unexpected)}")
    
    try:
        model.load_state_dict(state_dict, strict=True)
        print("[load] Strict load successful")
        return model
    except RuntimeError:
        print("[load] Strict load failed, trying flexible load...")
    
    new_state_dict = {}
    for ckpt_key, value in state_dict.items():
        if ckpt_key in model_keys:
            new_state_dict[ckpt_key] = value
        else:
            alternatives = [
                ckpt_key.replace("_cosine_head.", "head."),
                ckpt_key.replace("head.", "_cosine_head."),
            ]
            for alt_key in alternatives:
                if alt_key in model_keys:
                    new_state_dict[alt_key] = value
                    break
    
    model.load_state_dict(new_state_dict, strict=False)
    return model


# =============================================================================
# DATASET BUILDING
# =============================================================================

def build_eval_dataset(args_ns, class_to_idx, mean, std, label_col, is_multi_view):
    """
    Build evaluation dataset and return (dataset, original_dataframe).
    
    For multi-view, preserves image paths from both CSVs:
      - raw_img_path: 2.5x view (primary)
      - raw_img_path_10x: 10x view
    """
    if is_multi_view:
        csv_a = getattr(args_ns, "eval_csv_view1", None) or getattr(args_ns, "val_csv_view1", None)
        csv_b = getattr(args_ns, "eval_csv_view2", None) or getattr(args_ns, "val_csv_view2", None)
        assert csv_a and csv_b, "Multi-view eval requires --eval_csv_view1 and --eval_csv_view2"
        
        ds = PairedMultiFOV(
            csv_a=csv_a, csv_b=csv_b,
            class_to_idx=class_to_idx, size=args_ns.img_size, mean=mean, std=std,
            label_col=label_col, key_col="cell_id", slide_col="slide_id",
            augment=False, aug_mode="none", same_photometric=True
        )
        
        # Load original CSVs to preserve all columns including image paths
        df_a = pd.read_csv(csv_a)
        df_b = pd.read_csv(csv_b)
        
        key_col = "cell_id"
        df_b_subset = df_b[[key_col, 'raw_img_path']].copy()
        df_b_subset = df_b_subset.rename(columns={'raw_img_path': 'raw_img_path_10x'})
        
        original_df = df_a.merge(df_b_subset, on=key_col, how='inner')
        original_df = original_df[original_df[label_col].isin(class_to_idx.keys())].reset_index(drop=True)
        
        print(f"[info] Multi-view: preserved raw_img_path (2.5x) and raw_img_path_10x (10x)")
        
    else:
        csv_single = getattr(args_ns, "eval_csv", None) or getattr(args_ns, "val_csv", None)
        assert csv_single, "Single-view eval requires --eval_csv"
        
        ds = NucleusCSV(
            csv_single, class_to_idx, size=args_ns.img_size, mean=mean, std=std,
            label_col=label_col, augment=False, return_slide=True
        )
        
        original_df = pd.read_csv(csv_single)
        original_df = original_df[original_df[label_col].isin(class_to_idx.keys())].reset_index(drop=True)
    
    # Filter dataset
    before = len(ds.df)
    ds.df = ds.df[ds.df[label_col].isin(class_to_idx.keys())].reset_index(drop=True)
    removed = before - len(ds.df)
    if removed > 0:
        print(f"[filter] eval: removed {removed} rows not present in class_map")
    
    print(f"[info] Original CSV columns preserved: {len(original_df.columns)} columns")
    
    return ds, original_df


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser("CellPT/MorphPT Unified Eval")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .pt")
    
    # Dataset options
    p.add_argument("--eval_csv", type=str, default=None, help="Single-view CSV")
    p.add_argument("--eval_csv_view1", type=str, default=None, help="Multi-view: view-1 (2.5x) CSV")
    p.add_argument("--eval_csv_view2", type=str, default=None, help="Multi-view: view-2 (10x) CSV")
    p.add_argument("--label_col", type=str, default=None, help="Label column (default: from ckpt or 'cell_type')")
    
    # Eval options
    p.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--use_slide_mask", action="store_true", help="Apply slide-aware masking")
    
    # Logit adjustment
    p.add_argument("--apply_la", action="store_true", help="Apply logit adjustment at eval")
    p.add_argument("--tau", type=float, default=None, help="LA temperature")
    p.add_argument("--prior_csv", type=str, default=None, help="CSV with priors")
    
    # Output options
    p.add_argument("--out_dir", type=str, default=None, help="Output directory")
    p.add_argument("--save_confusions", action="store_true", help="Save confusion matrix PNGs")
    p.add_argument("--model_name", type=str, default=None, help="Model name for output")
    
    # Model override (for single-view baselines)
    p.add_argument("--backbone", type=str, default=None, help="Override backbone")
    p.add_argument("--force_single_view", action="store_true", 
                   help="Force single-view eval even if checkpoint is multi-view")
    
    return p.parse_args()


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

def main():
    args = parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = safe_load_checkpoint(args.ckpt)
    
    # Extract info from checkpoint
    classes = ckpt["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    ckpt_args = SimpleNamespace(**ckpt["args"]) if "args" in ckpt else SimpleNamespace()
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    
    # Determine if multi-view
    is_multi_view = int(getattr(ckpt_args, "multi_view", 0)) == 1
    if args.force_single_view:
        is_multi_view = False
        print("[info] Forcing single-view evaluation")
    
    # Merge CLI overrides
    if args.eval_csv: setattr(ckpt_args, "eval_csv", args.eval_csv)
    if args.eval_csv_view1: setattr(ckpt_args, "eval_csv_view1", args.eval_csv_view1)
    if args.eval_csv_view2: setattr(ckpt_args, "eval_csv_view2", args.eval_csv_view2)
    if args.batch_size: setattr(ckpt_args, "batch_size", args.batch_size)
    if not hasattr(ckpt_args, "img_size"): setattr(ckpt_args, "img_size", 224)
    
    label_col = args.label_col or getattr(ckpt_args, "label_col", "cell_type")
    
    # Output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        model_tag = args.model_name or ("mv" if is_multi_view else "sv")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        out_dir = Path(args.ckpt).parent / f"eval_{model_tag}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"Device: {device}, AMP dtype: {amp_dtype}")
    
    # ==========================================================================
    # BUILD MODEL
    # ==========================================================================
    if is_multi_view:
        print("\n[mode] MULTI-VIEW evaluation")
        # Use trainer's model builder for multi-view
        from trainer.main_trainer_refactor_v3 import build_backbone_and_heads, forward_batch
        
        model, mean, std, _, _ = build_backbone_and_heads(ckpt_args, classes, num_classes=num_classes, device=device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
    else:
        print("\n[mode] SINGLE-VIEW evaluation")
        # Infer backbone
        inferred_backbone, has_lora = infer_backbone_from_state_dict(state_dict)
        
        if args.backbone:
            backbone_name = args.backbone
        elif inferred_backbone:
            backbone_name = inferred_backbone
        elif hasattr(ckpt_args, 'backbone'):
            backbone_name = ckpt_args.backbone
        else:
            backbone_name = "resnet50"
        
        print(f"[info] Backbone: {backbone_name} (inferred={inferred_backbone}, has_lora={has_lora})")
        
        # Check for cosine classifier
        use_cosine = bool(getattr(ckpt_args, "cosine", 0))
        if "_cosine_head.weight" in state_dict or "_cosine_head._s_unconstrained" in state_dict:
            use_cosine = True
        
        # LoRA config
        lora_rank = int(getattr(ckpt_args, "lora_rank", 8))
        lora_alpha = float(getattr(ckpt_args, "lora_alpha", 2.0 * lora_rank))
        
        if has_lora:
            for k, v in state_dict.items():
                if "lora_down.weight" in k:
                    inferred_rank = v.shape[0]
                    if inferred_rank != lora_rank:
                        print(f"[lora] Rank mismatch: ckpt_args={lora_rank}, inferred={inferred_rank}. Using inferred.")
                        lora_rank = inferred_rank
                    break
        
        # Image size
        inferred_img_size = infer_image_size_from_state_dict(state_dict)
        img_size = int(getattr(ckpt_args, "img_size", inferred_img_size))
        setattr(ckpt_args, "img_size", img_size)
        print(f"[info] Image size: {img_size}")
        
        # Build model
        model, mean, std = build_sv_model(
            backbone_name, num_classes,
            use_cosine=use_cosine,
            has_lora=has_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            img_size=img_size,
            state_dict=state_dict,
        )
        
        model = load_weights_flexible(model, state_dict)
        model = model.to(device)
        model.eval()
    
    print(f"Model loaded. Classes: {num_classes}")
    
    # ==========================================================================
    # BUILD DATASET
    # ==========================================================================
    ds, original_df = build_eval_dataset(ckpt_args, class_to_idx, mean, std, label_col, is_multi_view)
    print(f"Eval samples: {len(ds)}")
    
    loader = DataLoader(
        ds,
        batch_size=int(getattr(ckpt_args, "batch_size", 64)),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn(getattr(ckpt_args, "seed", 42))
    )
    
    # ==========================================================================
    # OPTIONAL: SLIDE MASK & LOGIT ADJUSTMENT
    # ==========================================================================
    allowed_bitmask = build_allowed_bitmask_from_meta_json(classes, level="fine", subset="subset_all") \
                      if args.use_slide_mask else None
    
    adj = None
    if args.apply_la:
        prior_csv = args.prior_csv or getattr(ckpt_args, "prior_csv", None)
        if prior_csv:
            counts_np = class_counts_from_df(ds.df, label_col=label_col, classes=classes)
            prior_counts_np = load_priors_from_csv(prior_csv, label_col, class_to_idx, classes, counts_np)
            _, log_prior = compute_logit_prior(prior_counts_np, device)
            tau = args.tau if args.tau is not None else float(getattr(ckpt_args, "logit_tau", 1.0))
            adj = (tau * log_prior).view(1, -1)
            adj = adj - adj.mean()
            print(f"[LA] apply_la=1, tau={tau:.3f}")
        else:
            print("[LA] apply_la=1 requested, but no prior_csv available; skipping LA.")
    
    # Coarse taxonomy
    coarse_classes, coarse_to_idx, fine_to_coarse_idx, _ = \
        build_taxonomy_from_meta_json(classes, subset="subset_all", device=device)
    use_coarse_aux = bool(getattr(ckpt_args, "use_coarse_aux", 0))
    
    # ==========================================================================
    # EVALUATION LOOP
    # ==========================================================================
    all_preds, all_targets = [], []
    all_gates = []
    all_probs = []
    val_total = 0
    loss_sum = 0.0
    correct = 0
    
    print("\nRunning evaluation...")
    with torch.inference_mode(), torch.amp.autocast('cuda', dtype=amp_dtype, enabled=torch.cuda.is_available()):
        for batch_idx, batch in enumerate(loader):
            if len(batch) == 3:
                x, y, sid = batch
            else:
                x, y = batch
                sid = None
            
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            # Forward pass
            if is_multi_view:
                pre, _, aux = forward_batch(model, x, epoch=0, args=ckpt_args, amp_dtype=amp_dtype)
                if aux is not None and "g" in aux:
                    all_gates.append(aux["g"].detach().cpu())
                logits = aux["fused_logits"] if aux is not None else model._cosine_head(pre)
            else:
                logits = model(x)
                aux = None
            
            # Apply adjustments
            logits_eff = logits.clone()
            
            if adj is not None:
                logits_eff = logits_eff - adj.to(dtype=logits_eff.dtype, device=logits_eff.device)
            
            if use_coarse_aux and is_multi_view and hasattr(model, "coarse_head"):
                eta = float(getattr(ckpt_args, "hier_alpha", 0.2))
                if eta > 0.0:
                    eps = 1e-6
                    zc = model.coarse_head(pre)
                    qg = F.softmax(zc, dim=1).clamp_min(eps).log()
                    ftc = fine_to_coarse_idx.to(device)
                    if (ftc < 0).any():
                        ftc = ftc.clone()
                        ftc[ftc < 0] = 0
                        bonus = qg[:, ftc]
                        bonus[:, (fine_to_coarse_idx < 0)] = 0.0
                    else:
                        bonus = qg[:, ftc]
                    logits_eff = logits_eff + eta * bonus
            
            if args.use_slide_mask and allowed_bitmask is not None and sid is not None:
                logits_eff = apply_slide_mask(logits_eff, sid, allowed_bitmask)
            
            # Compute metrics
            loss = F.cross_entropy(logits_eff, y)
            loss_sum += loss.item() * y.size(0)
            
            probs = F.softmax(logits_eff, dim=1)
            preds = logits_eff.argmax(dim=1)
            correct += (preds == y).sum().item()
            
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
            all_probs.append(probs.cpu())
            val_total += y.size(0)
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {val_total} / {len(ds)} samples")
    
    if val_total == 0:
        print("No samples to evaluate. Check your CSV paths and label_col.")
        return
    
    # ==========================================================================
    # AGGREGATE RESULTS
    # ==========================================================================
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_probs = torch.cat(all_probs)
    
    if all_gates:
        all_gates = torch.cat(all_gates)
    else:
        all_gates = None
    
    accuracy = correct / val_total
    avg_loss = loss_sum / val_total
    
    # ==========================================================================
    # COMPUTE METRICS
    # ==========================================================================
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Samples:  {val_total}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Loss:     {avg_loss:.4f}")
    
    pc_metrics, _, _ = compute_per_class_metrics(all_preds, all_targets, classes)
    macro_f1 = float(pc_metrics.get("macro_f1", 0.0))
    print(f"Macro F1: {macro_f1:.4f}")
    
    print("\n" + "-" * 60)
    print("PER-CLASS METRICS")
    print("-" * 60)
    print_per_class_summary(pc_metrics, classes)
    
    # Coarse metrics
    coarse_metrics, _ = compute_coarse_metrics_from_fine(all_preds, all_targets, fine_to_coarse_idx, coarse_classes)
    print(f"\n[coarse] Overall Acc: {coarse_metrics['coarse_overall_acc']:.4f}, "
          f"Macro Recall: {coarse_metrics['coarse_macro_recall']:.4f}")
    
    # ==========================================================================
    # GATE ANALYSIS (multi-view only)
    # ==========================================================================
    if all_gates is not None:
        print("\n" + "=" * 60)
        print("GATE ANALYSIS (per-class average)")
        print("=" * 60)
        
        gate_df = pd.DataFrame(
            all_gates.numpy(),
            columns=[f"view_{i}" for i in range(all_gates.shape[1])]
        )
        gate_df["target"] = all_targets.numpy()
        gate_df["target_class"] = [classes[t] for t in all_targets.numpy()]
        gate_df.to_csv(out_dir / "gate_weights.csv", index=False)
        
        gate_summary = []
        for cls_idx, cls_name in enumerate(classes):
            mask = all_targets == cls_idx
            if mask.any():
                cls_gates = all_gates[mask].mean(dim=0)
                gate_str = " ".join([f"{g:.3f}" for g in cls_gates.tolist()])
                print(f"{cls_name:>30s}: [{gate_str}]")
                gate_summary.append({
                    "class": cls_name,
                    **{f"view_{i}": float(cls_gates[i]) for i in range(len(cls_gates))}
                })
        
        gate_summary_df = pd.DataFrame(gate_summary)
        gate_summary_df.to_csv(out_dir / "gate_summary_per_class.csv", index=False)
        print(f"Saved gate summary → {out_dir / 'gate_summary_per_class.csv'}")
    
    # ==========================================================================
    # SAVE PREDICTIONS WITH ORIGINAL DATA
    # ==========================================================================
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    results_df = original_df.copy()
    results_df["pred_idx"] = all_preds.numpy()
    results_df["true_idx"] = all_targets.numpy()
    results_df["pred_class"] = [classes[p] for p in all_preds.numpy()]
    results_df["true_class"] = [classes[t] for t in all_targets.numpy()]
    results_df["correct"] = (all_preds == all_targets).numpy()
    results_df["confidence"] = all_probs.max(dim=1).values.numpy()
    results_df["prob_pred_class"] = all_probs[torch.arange(len(all_preds)), all_preds].numpy()
    results_df["prob_true_class"] = all_probs[torch.arange(len(all_targets)), all_targets].numpy()
    
    if all_gates is not None:
        for v in range(all_gates.shape[1]):
            results_df[f"gate_view{v}"] = all_gates[:, v].numpy()
    
    results_df.to_csv(out_dir / "predictions_full.csv", index=False)
    print(f"Saved: {out_dir / 'predictions_full.csv'}")
    
    # Misclassified
    misclassified_df = results_df[~results_df["correct"]].copy()
    misclassified_df = misclassified_df.sort_values("confidence", ascending=False)
    misclassified_df.to_csv(out_dir / "misclassified.csv", index=False)
    print(f"Saved: {out_dir / 'misclassified.csv'} ({len(misclassified_df)} samples)")
    
    # High-confidence errors
    high_conf_errors = misclassified_df[misclassified_df["confidence"] > 0.8]
    if len(high_conf_errors) > 0:
        high_conf_errors.to_csv(out_dir / "high_confidence_errors.csv", index=False)
        print(f"Saved: {out_dir / 'high_confidence_errors.csv'} ({len(high_conf_errors)} samples)")
    
    # ==========================================================================
    # SAVE CONFUSION MATRICES
    # ==========================================================================
    if args.save_confusions:
        print("\nSaving confusion matrices...")
        save_confusion_matrices(
            all_targets.numpy(), all_preds.numpy(), classes, out_dir,
            epoch="eval", norms=("true", "none", "pred"),
            print_table_for="true", annotate=True
        )
    
    # ==========================================================================
    # SAVE METRICS JSON
    # ==========================================================================
    summary = {
        "model_name": args.model_name or ("multi-view" if is_multi_view else "single-view"),
        "is_multi_view": is_multi_view,
        "checkpoint": str(args.ckpt),
        "samples": val_total,
        "accuracy": accuracy,
        "loss": avg_loss,
        "macro_f1": macro_f1,
        "coarse_overall_acc": float(coarse_metrics["coarse_overall_acc"]),
        "coarse_macro_recall": float(coarse_metrics["coarse_macro_recall"]),
        "apply_la": bool(args.apply_la),
        "use_slide_mask": bool(args.use_slide_mask),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_dir / 'metrics_summary.json'}")
    
    pc_clean = json_serializable(pc_metrics)
    with open(out_dir / "per_class_metrics.json", "w") as f:
        json.dump(pc_clean, f, indent=2)
    print(f"Saved: {out_dir / 'per_class_metrics.json'}")
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY (for paper)")
    print("=" * 60)
    print(f"Model:    {args.model_name or ('MV' if is_multi_view else 'SV')}")
    print(f"Macro F1: {macro_f1:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print("=" * 60)
    
    print("\nDone.")


if __name__ == "__main__":
    main()