#!/usr/bin/env python3
"""
eval_sv_baselines.py - Evaluate single-view baseline checkpoints

Handles: ResNet-50, ConvNeXt, DINOv2 (with/without LoRA), Swin

Usage:
    python eval_sv_baselines.py \
        --ckpt /path/to/checkpoint.pt \
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
from data.dataset import NucleusCSV
from utils.utils import (
    worker_init_fn, 
    compute_per_class_metrics, 
    print_per_class_summary,
    save_confusion_matrices,
    build_taxonomy_from_meta_json,
    compute_coarse_metrics_from_fine,
)


def parse_args():
    p = argparse.ArgumentParser("Single-View Baseline Eval")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .pt")
    p.add_argument("--eval_csv", required=True, help="CSV file for evaluation")
    p.add_argument("--label_col", type=str, default=None, help="Label column (default: from ckpt or 'cell_type')")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--out_dir", type=str, default=None, help="Output directory")
    p.add_argument("--save_confusions", action="store_true", help="Save confusion matrix PNGs")
    p.add_argument("--model_name", type=str, default=None, help="Model name for output (e.g., 'ResNet-50')")
    p.add_argument("--backbone", type=str, default=None, 
                   help="Override backbone (e.g., 'vit_small_patch14_dinov2.lvd142m')")
    return p.parse_args()


class CosineClassifier(nn.Module):
    """Cosine similarity classifier with learnable temperature."""
    def __init__(self, in_features, num_classes, init_temp=0.07):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        # Support both naming conventions
        self._s_unconstrained = nn.Parameter(torch.tensor(1.0 / init_temp))
    
    @property
    def scale(self):
        return self._s_unconstrained
    
    def forward(self, x):
        x_norm = F.normalize(x, dim=-1)
        w_norm = F.normalize(self.weight, dim=-1)
        return self._s_unconstrained * (x_norm @ w_norm.T)


def get_lora_target_modules_from_state_dict(state_dict):
    """
    Extract which modules have LoRA adapters from checkpoint keys.
    Returns a list of full module paths that should have LoRA.
    """
    lora_modules = set()
    for key in state_dict.keys():
        if ".lora_down.weight" in key:
            # Extract the base module path
            # e.g., "blocks.8.attn.qkv.lora_down.weight" -> "blocks.8.attn.qkv"
            base_path = key.replace(".lora_down.weight", "")
            lora_modules.add(base_path)
    return list(lora_modules)


class LoRALinear(nn.Module):
    """Linear layer with LoRA adapters that matches checkpoint naming."""
    def __init__(self, in_features, out_features, rank=8, alpha=None, bias=True):
        super().__init__()
        # Base linear (will be loaded from checkpoint)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        
        # LoRA adapters - use nn.Linear to match checkpoint key structure
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        
        # Initialize LoRA
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # LoRA scale: alpha / rank (default alpha = 2*rank for scale=2.0)
        if alpha is None:
            alpha = 2.0 * rank  # Default to scale=2.0
        self.scale = alpha / rank
    
    def forward(self, x):
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = self.lora_up(self.lora_down(x)) * self.scale
        return base_out + lora_out


def inject_lora_from_checkpoint(model, state_dict, default_rank=8, alpha=None):
    """
    Inject LoRA adapters into model at exactly the locations specified in checkpoint.
    
    Args:
        model: The base model
        state_dict: Checkpoint state dict
        default_rank: Default LoRA rank if not inferable
        alpha: LoRA alpha for scaling (scale = alpha/rank). If None, defaults to 2*rank.
    """
    lora_modules = get_lora_target_modules_from_state_dict(state_dict)
    
    if not lora_modules:
        print("[lora] No LoRA modules found in checkpoint")
        return model
    
    print(f"[lora] Found {len(lora_modules)} LoRA modules in checkpoint")
    
    # Infer rank from checkpoint
    rank = default_rank
    for key in state_dict.keys():
        if ".lora_down.weight" in key:
            rank = state_dict[key].shape[0]
            break
    
    # Calculate scale
    if alpha is None:
        alpha = 2.0 * rank  # Default scale = 2.0
    scale = alpha / rank
    print(f"[lora] Inferred rank={rank}, alpha={alpha}, scale={scale:.2f}")
    
    # Replace each target module with LoRA version
    for module_path in lora_modules:
        parts = module_path.split(".")
        
        # Navigate to parent module
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        
        # Get the target module
        attr_name = parts[-1]
        try:
            old_module = getattr(parent, attr_name)
        except AttributeError:
            print(f"[lora] Warning: Could not find {module_path}")
            continue
        
        if not isinstance(old_module, nn.Linear):
            print(f"[lora] Warning: {module_path} is not nn.Linear, skipping")
            continue
        
        # Create LoRA replacement with proper alpha
        new_module = LoRALinear(
            in_features=old_module.in_features,
            out_features=old_module.out_features,
            rank=rank,
            alpha=alpha,
            bias=old_module.bias is not None,
        )
        
        # Copy base weights
        new_module.weight.data.copy_(old_module.weight.data)
        if old_module.bias is not None:
            new_module.bias.data.copy_(old_module.bias.data)
        
        # Replace
        setattr(parent, attr_name, new_module)
    
    print(f"[lora] Injected LoRA into {len(lora_modules)} modules")
    return model


def infer_backbone_from_state_dict(state_dict):
    """Infer the backbone architecture from checkpoint state_dict keys."""
    keys = list(state_dict.keys())
    
    # DINOv2 / ViT detection
    if any("cls_token" in k for k in keys) or any("patch_embed" in k for k in keys):
        # More specific LoRA detection - look for actual LoRA adapter keys
        has_lora = any(".lora_down." in k or ".lora_up." in k for k in keys)
        
        # Try to determine size from embedding dimension
        if "patch_embed.proj.weight" in keys:
            embed_dim = state_dict["patch_embed.proj.weight"].shape[0]
        elif any("head.weight" in k for k in keys):
            embed_dim = state_dict["head.weight"].shape[1] if "head.weight" in keys else None
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
    
    # Swin detection
    if any("layers.0.blocks" in k for k in keys):
        return "swin_base_patch4_window7_224", False
    
    # ConvNeXt detection  
    if any("stages" in k for k in keys) or any("downsample_layers" in k for k in keys):
        return "convnext_base", False
    
    # ResNet detection
    if any("layer1" in k for k in keys) and any("conv1" in k for k in keys):
        return "resnet50", False
    
    return None, False


def infer_image_size_from_state_dict(state_dict):
    """
    Infer the training image size from pos_embed shape.
    For ViT with patch_size=14: n_patches = (img_size/14)^2
    pos_embed shape = [1, n_patches + 1, embed_dim]  (the +1 is for CLS token)
    """
    if "pos_embed" not in state_dict:
        return 224  # default
    
    pos_embed_shape = state_dict["pos_embed"].shape
    n_tokens = pos_embed_shape[1]  # includes CLS token
    n_patches = n_tokens - 1
    
    # For DINOv2 with patch_size=14
    patches_per_side = int(round(n_patches ** 0.5))
    if patches_per_side * patches_per_side == n_patches:
        img_size = patches_per_side * 14
        print(f"[info] Inferred img_size={img_size} from pos_embed shape {pos_embed_shape}")
        return img_size
    
    return 224  # fallback


def build_model(backbone_name, num_classes, use_cosine=True, has_lora=False, lora_rank=8, lora_alpha=None, img_size=224, state_dict=None):
    """Build model with proper architecture."""
    import timm
    
    # Default alpha to 2*rank (scale=2.0) if not specified
    if lora_alpha is None:
        lora_alpha = 2.0 * lora_rank
    
    print(f"[model] Building {backbone_name}, cosine={use_cosine}, lora={has_lora}, img_size={img_size}")
    if has_lora:
        print(f"[model] LoRA config: rank={lora_rank}, alpha={lora_alpha}, scale={lora_alpha/lora_rank:.2f}")
    
    # DINOv2 models via timm
    if "dinov2" in backbone_name.lower() or "vit" in backbone_name.lower():
        # timm model names for DINOv2
        if "small" in backbone_name.lower() or "vits" in backbone_name.lower():
            timm_name = "vit_small_patch14_dinov2.lvd142m"
            embed_dim = 384
        elif "base" in backbone_name.lower() or "vitb" in backbone_name.lower():
            timm_name = "vit_base_patch14_dinov2.lvd142m"
            embed_dim = 768
        elif "large" in backbone_name.lower() or "vitl" in backbone_name.lower():
            timm_name = "vit_large_patch14_dinov2.lvd142m"
            embed_dim = 1024
        else:
            timm_name = backbone_name
            embed_dim = 384  # default
        
        # Create model with correct image size
        model = timm.create_model(
            timm_name, 
            pretrained=False, 
            num_classes=0,  # No head
            img_size=img_size,  # Match checkpoint
        )
        
        # Inject LoRA if checkpoint has LoRA weights
        if has_lora and state_dict is not None:
            model = inject_lora_from_checkpoint(model, state_dict, default_rank=lora_rank, alpha=lora_alpha)
        
        # Add classification head
        if use_cosine:
            model._cosine_head = CosineClassifier(embed_dim, num_classes)
            # Also add regular head for compatibility
            model.head = model._cosine_head
        else:
            model.head = nn.Linear(embed_dim, num_classes)
        
        # Override forward to handle ViT feature extraction properly
        def new_forward(x):
            features = model.forward_features(x)
            # forward_features returns [B, num_tokens, embed_dim] for ViT
            # We need just the CLS token: [B, embed_dim]
            if features.dim() == 3:
                features = features[:, 0]  # Take CLS token (index 0)
            if hasattr(model, '_cosine_head'):
                return model._cosine_head(features)
            return model.head(features)
        model.forward = new_forward
        
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
    elif "convnext" in backbone_name.lower():
        model = timm.create_model(backbone_name, pretrained=False, num_classes=num_classes)
        if use_cosine:
            in_features = model.head.fc.in_features if hasattr(model.head, 'fc') else model.head.in_features
            if hasattr(model.head, 'fc'):
                model.head.fc = CosineClassifier(in_features, num_classes)
            else:
                model.head = CosineClassifier(in_features, num_classes)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
    elif "swin" in backbone_name.lower():
        model = timm.create_model(backbone_name, pretrained=False, num_classes=num_classes)
        if use_cosine:
            in_features = model.head.fc.in_features if hasattr(model.head, 'fc') else model.head.in_features
            if hasattr(model.head, 'fc'):
                model.head.fc = CosineClassifier(in_features, num_classes)
            else:
                model.head = CosineClassifier(in_features, num_classes)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
    else:  # ResNet and others
        model = timm.create_model(backbone_name, pretrained=False, num_classes=num_classes)
        if use_cosine and hasattr(model, 'fc'):
            in_features = model.fc.in_features
            model.fc = CosineClassifier(in_features, num_classes)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    
    return model, mean, std


def load_weights_flexible(model, state_dict):
    """
    Load weights with flexible key matching.
    Handles mismatches between checkpoint and model keys.
    """
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    
    # Find common keys
    common_keys = model_keys & ckpt_keys
    missing_in_ckpt = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys
    
    print(f"[load] Common keys: {len(common_keys)}, Missing: {len(missing_in_ckpt)}, Unexpected: {len(unexpected)}")
    
    # Try direct load first
    try:
        model.load_state_dict(state_dict, strict=True)
        print("[load] Strict load successful")
        return model
    except RuntimeError as e:
        print(f"[load] Strict load failed, trying flexible load...")
    
    # Build mapping for mismatched keys
    new_state_dict = {}
    
    for ckpt_key, value in state_dict.items():
        if ckpt_key in model_keys:
            new_state_dict[ckpt_key] = value
        else:
            # Try to find matching key in model
            # Handle common naming variations
            alternatives = [
                ckpt_key,
                ckpt_key.replace("_cosine_head.", "head."),
                ckpt_key.replace("head.", "_cosine_head."),
                ckpt_key.replace(".lora_down.weight", ".lora_down.weight"),
                ckpt_key.replace(".lora_up.weight", ".lora_up.weight"),
            ]
            
            matched = False
            for alt_key in alternatives:
                if alt_key in model_keys:
                    new_state_dict[alt_key] = value
                    matched = True
                    break
            
            if not matched and "lora" not in ckpt_key:
                # Only warn for non-LoRA keys
                pass  # print(f"[load] Skipping unmatched key: {ckpt_key}")
    
    # Load what we can
    model.load_state_dict(new_state_dict, strict=False)
    
    # Check what's still missing
    final_missing = set(model.state_dict().keys()) - set(new_state_dict.keys())
    if final_missing:
        # Filter out expected missing keys (like num_batches_tracked)
        important_missing = [k for k in final_missing if "num_batches_tracked" not in k]
        if important_missing:
            print(f"[load] Warning: {len(important_missing)} keys not loaded")
            if len(important_missing) <= 10:
                for k in important_missing:
                    print(f"  - {k}")
    
    return model


def main():
    args = parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    
    # Extract info from checkpoint
    classes = ckpt["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    ckpt_args = SimpleNamespace(**ckpt["args"]) if "args" in ckpt else SimpleNamespace()
    
    # Get state dict
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    
    # Determine backbone
    inferred_backbone, has_lora = infer_backbone_from_state_dict(state_dict)
    
    # Priority: CLI arg > inferred > checkpoint args > default
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
    
    # LoRA rank and alpha (try to infer from checkpoint args)
    lora_rank = int(getattr(ckpt_args, "lora_rank", 8))
    lora_alpha = float(getattr(ckpt_args, "lora_alpha", 2.0 * lora_rank))  # Default scale=2.0
    
    if has_lora:
        # Try to infer rank from weight shapes (overrides ckpt_args if different)
        for k, v in state_dict.items():
            if "lora_down.weight" in k:
                inferred_rank = v.shape[0]
                if inferred_rank != lora_rank:
                    print(f"[lora] Rank mismatch: ckpt_args={lora_rank}, inferred={inferred_rank}. Using inferred.")
                    lora_rank = inferred_rank
                break
    
    # Label column
    label_col = args.label_col or getattr(ckpt_args, "label_col", "cell_type")
    
    # Output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        model_tag = args.model_name or backbone_name.replace("/", "_").replace(".", "_")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        out_dir = Path(args.ckpt).parent / f"eval_{model_tag}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"Device: {device}, AMP dtype: {amp_dtype}")
    
    # Infer image size from checkpoint (for ViT models)
    inferred_img_size = infer_image_size_from_state_dict(state_dict)
    img_size = int(getattr(ckpt_args, "img_size", inferred_img_size))
    print(f"[info] Image size: {img_size} (inferred={inferred_img_size})")
    
    # Build model
    model, mean, std = build_model(
        backbone_name, num_classes, 
        use_cosine=use_cosine, 
        has_lora=has_lora, 
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        img_size=img_size,
        state_dict=state_dict,
    )
    
    # Load weights
    model = load_weights_flexible(model, state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded. Classes: {num_classes}")
    
    # Build dataset
    img_size = int(getattr(ckpt_args, "img_size", 224))
    ds = NucleusCSV(
        args.eval_csv,
        class_to_idx,
        size=img_size,
        mean=mean,
        std=std,
        label_col=label_col,
        augment=False,
        return_slide=True
    )
    
    # Filter to known classes
    before = len(ds.df)
    ds.df = ds.df[ds.df[label_col].isin(class_to_idx.keys())].reset_index(drop=True)
    removed = before - len(ds.df)
    if removed > 0:
        print(f"[filter] Removed {removed} rows with unknown classes")
    print(f"Eval samples: {len(ds)}")
    
    # Keep original df for saving predictions
    original_df = ds.df.copy()
    
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn(42)
    )
    
    # Eval loop
    all_preds = []
    all_targets = []
    all_probs = []
    total_loss = 0.0
    correct = 0
    total = 0
    
    print("\nRunning evaluation...")
    with torch.inference_mode(), torch.amp.autocast('cuda', dtype=amp_dtype, enabled=torch.cuda.is_available()):
        for batch_idx, batch in enumerate(loader):
            if len(batch) == 3:
                x, y, sid = batch
            else:
                x, y = batch
            
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            logits = model(x)
            
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item() * y.size(0)
            
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
            all_probs.append(probs.cpu())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {total} / {len(ds)} samples")
    
    # Aggregate results
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_probs = torch.cat(all_probs)
    
    accuracy = correct / total
    avg_loss = total_loss / total
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Samples:  {total}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Loss:     {avg_loss:.4f}")
    
    # Per-class metrics
    pc_metrics, _, _ = compute_per_class_metrics(all_preds, all_targets, classes)
    macro_f1 = float(pc_metrics.get("macro_f1", 0.0))
    print(f"Macro F1: {macro_f1:.4f}")
    
    print("\n" + "-" * 60)
    print("PER-CLASS METRICS")
    print("-" * 60)
    print_per_class_summary(pc_metrics, classes)
    
    # Coarse metrics
    try:
        coarse_classes, coarse_to_idx, fine_to_coarse_idx, _ = \
            build_taxonomy_from_meta_json(classes, subset="subset_all", device=device)
        coarse_metrics, _ = compute_coarse_metrics_from_fine(
            all_preds, all_targets, fine_to_coarse_idx, coarse_classes
        )
        print(f"\n[coarse] Overall Acc: {coarse_metrics['coarse_overall_acc']:.4f}, "
              f"Macro Recall: {coarse_metrics['coarse_macro_recall']:.4f}")
    except Exception as e:
        print(f"[warn] Could not compute coarse metrics: {e}")
        coarse_metrics = {}
    
    # ================================================================
    # SAVE FULL PREDICTIONS WITH ORIGINAL DATA
    # ================================================================
    print("\nSaving predictions...")
    
    # Merge predictions with original data
    pred_df = original_df.copy()
    pred_df["pred_idx"] = all_preds.numpy()
    pred_df["target_idx"] = all_targets.numpy()
    pred_df["pred_class"] = [classes[p] for p in all_preds.numpy()]
    pred_df["true_class"] = [classes[t] for t in all_targets.numpy()]
    pred_df["correct"] = (all_preds == all_targets).numpy()
    pred_df["confidence"] = all_probs.max(dim=1).values.numpy()
    
    pred_df.to_csv(out_dir / "predictions_full.csv", index=False)
    print(f"Saved: {out_dir / 'predictions_full.csv'}")
    
    # Save misclassified
    misclassified = pred_df[~pred_df["correct"]].sort_values("confidence", ascending=False)
    misclassified.to_csv(out_dir / "misclassified.csv", index=False)
    print(f"Saved: {out_dir / 'misclassified.csv'} ({len(misclassified)} samples)")
    
    # ================================================================
    # SAVE CONFUSION MATRICES
    # ================================================================
    if args.save_confusions:
        print("\nSaving confusion matrices...")
        save_confusion_matrices(
            all_targets.numpy(), all_preds.numpy(), classes, out_dir,
            epoch="eval", norms=("true", "none", "pred"), 
            print_table_for="true", annotate=True
        )
    
    # ================================================================
    # SAVE METRICS
    # ================================================================
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
    
    pc_metrics_clean = json_serializable(pc_metrics)
    with open(out_dir / "per_class_metrics.json", "w") as f:
        json.dump(pc_metrics_clean, f, indent=2)
    
    summary = {
        "model_name": args.model_name or backbone_name,
        "backbone": backbone_name,
        "has_lora": has_lora,
        "use_cosine": use_cosine,
        "checkpoint": str(args.ckpt),
        "eval_csv": str(args.eval_csv),
        "samples": total,
        "accuracy": accuracy,
        "loss": avg_loss,
        "macro_f1": macro_f1,
        "coarse_overall_acc": float(coarse_metrics.get("coarse_overall_acc", 0)),
        "coarse_macro_recall": float(coarse_metrics.get("coarse_macro_recall", 0)),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "classes": classes,
    }
    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print("\n" + "=" * 60)
    print("SUMMARY (for paper)")
    print("=" * 60)
    print(f"Model:    {args.model_name or backbone_name}")
    print(f"Macro F1: {macro_f1:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print("=" * 60)
    
    print("\nDone.")


if __name__ == "__main__":
    main()