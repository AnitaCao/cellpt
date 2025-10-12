#!/usr/bin/env python3
import os, json, math, time, warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score

import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.utils import ModelEmaV2

# project imports (unchanged)
from config.opts import parse_args, add_lora_args
from data.dataset import NucleusCSV
from utils.utils import (
    set_seed, compute_metrics,
    init_metrics_logger, log_metrics, compute_per_class_metrics,
    print_per_class_summary, class_counts_from_df, maybe_init_wandb, 
    print_confusion_slice, save_confusion_matrices,
    worker_init_fn, find_group_lr, apply_slide_mask,
    evaluate_with_mask, log_to_wandb, MacroF1EarlyStopping,
    detailed_prediction_analysis, build_allowed_bitmask,
    setup_mixup, create_scheduler, build_optimizer,
    load_priors_from_csv, compute_logit_prior, tau_at_epoch,
    build_dataloaders, build_allowed_bitmask_from_meta_json,build_taxonomy_from_meta_json,
    compute_coarse_metrics_from_fine,
    save_coarse_confusions_from_fine,
)
from utils.losses import CBFocalLoss, LDAMLoss

from model.lora import apply_lora_to_timm_vit



# -----------------------------
# Cosine head with learnable temperature
# -----------------------------
class CosineHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, s_init: float = 30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")
        self._s_unconstrained = nn.Parameter(torch.tensor(s_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        s = F.softplus(self._s_unconstrained) + 1e-6  # ensure s > 0
        return s * (x @ w.t())


def replace_classifier_with_cosine(model: nn.Module, num_classes: int, s_init: float = 30.0) -> CosineHead:
    """
    Replace the backbone's classifier layer with a CosineHead in a model-agnostic way.
    Returns the CosineHead module (so we can build optim groups reliably).
    """
    # 1) infer input dim from current classifier if possible
    in_dim = None
    try:
        clf = model.get_classifier()
        if hasattr(clf, "in_features"):
            in_dim = clf.in_features
    except Exception:
        pass
    if in_dim is None:
        in_dim = getattr(model, "num_features", None)
    if in_dim is None:
        raise RuntimeError("Could not determine classifier in_features for this backbone.")

    cos_head = CosineHead(in_dim, num_classes, s_init=s_init)

    # 2) find classifier path from default_cfg if present
    name = None
    try:
        name = model.default_cfg.get("classifier", None)
    except Exception:
        name = None

    def _set_by_path(root, path: str, mod):
        obj = root
        parts = path.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], mod)

    replaced = False
    if name:
        try:
            _set_by_path(model, name, cos_head)
            replaced = True
        except Exception:
            replaced = False

    # 3) fallbacks
    if not replaced:
        if hasattr(model, "fc"):
            model.fc = cos_head
            replaced = True
        elif hasattr(model, "classifier"):
            model.classifier = cos_head
            replaced = True
        elif hasattr(model, "head") and isinstance(getattr(model, "head"), nn.Linear):
            model.head = cos_head
            replaced = True

    if not replaced:
        # Last resort: try reset_classifier if available, then set by attr
        try:
            model.reset_classifier(num_classes=num_classes)
        except Exception:
            pass
        # try again quickly
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            model.fc = cos_head
            replaced = True
        elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
            model.classifier = cos_head
            replaced = True
        elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
            model.head = cos_head
            replaced = True

    if not replaced:
        raise RuntimeError("Failed to replace classifier with CosineHead for this backbone.")

    # Save a stable alias; DO NOT overwrite the model's structural 'head' module (e.g., ConvNeXt head)
    setattr(model, "_cosine_head", cos_head)
    return cos_head


def masked_gt_rate(sids, y, mask):
    bad = 0
    for sid, yi in zip(sids, y.tolist()):
        m = mask.get(sid)
        if m is not None and not bool(m[yi].item()):
            bad += 1
    return bad / max(1, len(y))

def main():
    args = parse_args(add_lora_args)

    print("Training DINOv2 (or other timm backbone) with LoRA (ViT only) and Cosine Head")
    print("==> Launch config")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # ---- Env toggles ----
 
    # EMA
    ema_on      = os.environ.get("EMA", "0") == "1"
    ema_decay   = float(os.environ.get("EMA_DECAY", "0.9999"))
    ema_eval    = os.environ.get("EMA_EVAL", "1") == "1"
    # Data shaping
    smooth_on       = os.environ.get("SMOOTH_SAMPLER", "0") == "1"
    smooth_alpha    = float(os.environ.get("SMOOTH_ALPHA", "0.5"))   # 0.5 ~ sqrt
    epoch_budget    = int(os.environ.get("EPOCH_BUDGET", "0"))
    balanced_last_k = int(os.environ.get("BALANCED_LAST_K", "0"))           # e.g., 3


    worker_init = worker_init_fn(args.seed)

    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load class map first ----
    with open(args.class_map, "r") as f:
        class_to_idx = json.load(f)
    classes = sorted(class_to_idx, key=lambda k: class_to_idx[k])
    
    
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")

    # sanity: contiguous 0..K-1 indices
    idxs = sorted(class_to_idx.values())
    if idxs != list(range(len(idxs))):
        warnings.warn(f"[label-map] Indices are not contiguous 0..{len(idxs)-1}. Got: {idxs[:min(10,len(idxs))]} ...")
        
        
    coarse_classes, coarse_to_idx, fine_to_coarse_idx, fine_to_coarse_name = \
        build_taxonomy_from_meta_json(classes, subset="subset_all", device=device)
        
    print(f"Coarse classes ({len(coarse_classes)}): {coarse_classes}")
    print(f"Coarse mapping example: {list(fine_to_coarse_name.items())[:]} ...")

    # ---- Build model ----
    model = timm.create_model(
        args.model,
        pretrained=True,
        num_classes=num_classes,  # will be replaced by cosine head immediately
        img_size=args.img_size
    )

    # Get normalization from model cfg (fallback to ImageNet defaults)
    try:
        mean = tuple(model.pretrained_cfg.get('mean', IMAGENET_DEFAULT_MEAN))
        std  = tuple(model.pretrained_cfg.get('std',  IMAGENET_DEFAULT_STD))
    except Exception:
        mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    # Freeze or unfreeze
    unfreeze = bool(getattr(args, "unfreeze_backbone", False))
    if unfreeze:
        for p in model.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = False

    # Replace classifier with cosine head (backbone-agnostic), then unfreeze head params
    cos_head = replace_classifier_with_cosine(model, num_classes, s_init=30.0)
    
    
    assert num_classes == cos_head.weight.size(0), "Head out_dim does not match class map"
    for p in cos_head.parameters():
        p.requires_grad = True


    with torch.no_grad():
        target_s = torch.tensor(30.0)
        cos_head._s_unconstrained.copy_(torch.log(torch.expm1(target_s)))
    cos_head._s_unconstrained.requires_grad = True

    # LoRA (ViT only; skipped for ConvNeXt / Swin)
    is_vit = hasattr(model, "blocks") and len(getattr(model, "blocks", [])) > 0
    lora_params: List[torch.nn.Parameter] = []
    if not unfreeze and is_vit and getattr(args, "lora_blocks", 0) > 0:
        lora_params = apply_lora_to_timm_vit(
            model,
            last_n_blocks=args.lora_blocks,
            r=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )

    # Unfreeze norms in the last args.lora_blocks (ViT only)
    norms = []
    if not unfreeze and is_vit and getattr(args, "lora_blocks", 0) > 0:
        total_blocks = len(model.blocks)
        start = max(0, total_blocks - args.lora_blocks)
        for i in range(start, total_blocks):
            for name in ("norm1", "norm2"):
                m = getattr(model.blocks[i], name, None)
                if m is not None:
                    for p in m.parameters():
                        p.requires_grad = True
                    norms += list(m.parameters())

    model.to(device)
    
    with torch.no_grad():
        _probe = torch.randn(2, 3, args.img_size, args.img_size, device=device, dtype=next(model.parameters()).dtype)
        _out = model(_probe)
        assert _out.shape[-1] == num_classes, "Cosine head replacement failed; output dims do not match num_classes."
    

    try:
        n_lora = sum(p.numel() for p in lora_params)
        if n_lora > 0:
            print(f"LoRA trainable params: {n_lora}")
    except Exception:
        pass
    

    # -------- Mixup / CutMix wiring --------    
    mixup_enabled, mixup_fn, soft_ce, mixup_off_k = setup_mixup(args, num_classes)


    label_col = getattr(args, "label_col", "cell_type")
    print(f"Using label column: {label_col}")
    print(f"Base sampler policy: {args.sampler.upper()}")

    # ---- Datasets ----
    train_ds = NucleusCSV(
        args.train_csv, class_to_idx, size=args.img_size, mean=mean, std=std,
        label_col=label_col, augment=True,
        return_slide=True
    )
    val_ds = NucleusCSV(
        args.val_csv, class_to_idx, size=args.img_size, mean=mean, std=std,
        label_col=label_col, augment=False,
        return_slide=True
    )
    
    
    for ds_name, ds in (("train", train_ds), ("val", val_ds)):
        before = len(ds.df)
        ds.df = ds.df[ds.df[label_col].isin(class_to_idx.keys())].reset_index(drop=True)
        removed = before - len(ds.df)
        if removed > 0:
            print(f"[filter] {ds_name}: removed {removed} rows whose labels are not in class_map")

    print(f"Train samples: {len(train_ds)}")
    if unfreeze:
        print(f"Full finetune mode: unfreeze_backbone=1  lr_backbone={args.lr_backbone}")
    print(f"Val samples:   {len(val_ds)}")
    

    # Show class distribution (aligned to head order)
    counts_np = class_counts_from_df(train_ds.df, label_col=label_col, classes=classes)
    total = max(1, counts_np.sum())
    print("Train class distribution:")
    for cls, c in zip(classes, counts_np):
        pct = 100.0 * float(c) / float(total)
        print(f"  {cls:>24}: {int(c):7d} ({pct:5.1f}%)")
        
    
    if args.loss_type in ["cb_focal", "ldam_drw"] and args.sampler == "weighted":
        print("[warning] Switching sampler to UNIFORM because CB-Focal/LDAM-DRW do not pair well with weighted base sampling.")
        args.sampler = "uniform"

    # ---- DataLoaders ----
    loaders = build_dataloaders(
        train_ds=train_ds,
        val_ds=val_ds,
        args=args,
        classes=classes,
        label_col=label_col,
        worker_init_fn=worker_init,
        smooth_on=smooth_on,
        smooth_alpha=smooth_alpha,
        epoch_budget=epoch_budget,
        balanced_last_k=balanced_last_k,
        device=device,
    )

    train_loader_shuffle  = loaders["train_shuffle"]
    train_loader_weighted = loaders["train_weighted"]
    train_loader_smooth   = loaders["train_smooth"]
    balanced_loader       = loaders["train_balanced"]
    val_loader            = loaders["val"]
    
    #Build slide‑level allowed class bitmasks for constrained inference
    #allowed_bitmask = build_allowed_bitmask(val_ds.df, classes, label_col=label_col, slide_col="slide_id")
    
    allowed_bitmask = build_allowed_bitmask_from_meta_json(classes, level="fine", subset="subset_all")

    print(f"Built context masks for {len(allowed_bitmask)} slides")


    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16))

    # ---- Priors / LA ----
    prior_counts_np = load_priors_from_csv(getattr(args, "prior_csv", None),
                                       label_col, class_to_idx, classes, counts_np)
    _, log_prior = compute_logit_prior(prior_counts_np, device)


    early_stop_off = os.environ.get("EARLY_STOP_OFF", "0") == "1"
    tau_target = float(getattr(args, "logit_tau", 1.0))
    if os.environ.get("DISABLE_LA", "0") == "1":
        tau_target = 0.0
    la_start_epoch = int(getattr(args, "la_start_epoch", 3))
    la_ramp_epochs = int(getattr(args, "la_ramp_epochs", 3))

    print(f"DISABLE_LA={'1' if tau_target==0.0 else '0'}  EARLY_STOP_OFF={'1' if early_stop_off else '0'}")
    if smooth_on:
        eb_text = epoch_budget if epoch_budget > 0 else "full"
        print(f"Power smoothing enabled: alpha={smooth_alpha:.3f}  epoch_budget={eb_text}")
    print(f"\nLogit adjustment configured: tau_target={tau_target:.2f}, "
          f"la_start_epoch={la_start_epoch}, la_ramp_epochs={la_ramp_epochs}")

    # Default label smoothing (used when mixup is OFF)
    ls_target = float(getattr(args, "label_smoothing", 0.0))


    # ---- Imbalance-aware loss selection ----
    cls_counts = counts_np.astype(np.int64) 
    
    cb_loss = None
    ldam_loss = None
    if args.loss_type == "cb_focal":
        cb_loss = CBFocalLoss(cls_counts, beta=args.cb_beta, gamma=args.focal_gamma).to(device)
    elif args.loss_type == "ldam_drw":
        ldam_loss = LDAMLoss(cls_counts, max_m=args.ldam_max_m, s=args.ldam_s,
                             drw=True, beta=args.cb_beta, drw_start=args.drw_start_epoch).to(device)

    '''
    # Mixup only with CE
    if args.loss_type != "ce" and mixup_enabled:
        print("[loss] Disabling mixup/cutmix since loss_type != 'ce'.")
        mixup_enabled = False
        mixup_fn = None
        soft_ce = None
    '''
    # Disable LA for non‑CE losses
    if args.loss_type != "ce" and tau_target > 0.0:
        print("[LA] Disabled for non‑CE loss.")
        tau_target = 0.0



    head_weight_params, head_scale_params = [], []
    for n, p in getattr(model, "_cosine_head").named_parameters():
        if not p.requires_grad:
            continue
        if n == "_s_unconstrained":
            head_scale_params.append(p)
        else:
            head_weight_params.append(p)
            
    
    # ---- Coarse aux head (guarded by flag) ----
    use_coarse_aux = bool(getattr(args, "use_coarse_aux", 0))
    if use_coarse_aux:
        with torch.inference_mode():
            _probe = torch.randn(2, 3, args.img_size, args.img_size,
                                device=device, dtype=next(model.parameters()).dtype)
            feats = model.forward_features(_probe)
        in_dim = feats.shape[-1]
        model.coarse_head = nn.Linear(in_dim, len(coarse_classes)).to(device)

    # Optimizer groups (coarse head gets head lr/WD)
    optimizer = build_optimizer(model, args, lora_params, norms, head_weight_params, head_scale_params)
    

    # EMA (track full model)
    ema_model = ModelEmaV2(model, decay=ema_decay, device=device) if ema_on else None
    if ema_model is not None:
        print(f"[EMA] Enabled (decay={ema_decay}, eval={'EMA' if ema_eval else 'student'})")

    # LR scheduler
    scheduler = create_scheduler(args, optimizer)


    # Persist class map used
    with open(Path(args.out_dir, "class_to_idx.used.json"), "w") as f:
        json.dump(class_to_idx, f, indent=2)
    
    run_id = time.strftime("%Y%m%d-%H%M%S")

    # Metrics + W&B
    best_metric = -1.0
    #create path file name based on model and settings
    file_name = f"{args.model}_bs{args.batch_size}_lrh{args.lr_head}_{run_id}.pt"
    best_path = str(Path(args.out_dir, file_name))
    save_by = getattr(args, "save_metric", "acc")  # "acc" or "f1"
    
    metrics_f, metrics_w = init_metrics_logger(args.out_dir, filename=f"metrics_{run_id}.csv", classes=classes)
    wandb_mod, wb_run = maybe_init_wandb(args, classes, run_id, args.out_dir)

    print("\nStarting training (single GPU)...")
    print("Using cosine head and " + ("logit-adjusted CE (with ramp)" if tau_target>0 else "plain CE (no LA)"))
    print(f"LRs: head={args.lr_head}  lora={getattr(args,'lr_lora',0.0)}  | cosine_schedule={args.cosine} warmup={args.warmup_epochs}\n")

    params_to_clip = [p for g in optimizer.param_groups for p in g['params'] if p.requires_grad]
    
    early_stopper = MacroF1EarlyStopping(patience=20, min_delta=0.002)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Mixup schedule
        use_mixup_this_epoch = mixup_enabled and args.loss_type == "ce" and (epoch <= (args.epochs - max(0, mixup_off_k)))
        if mixup_enabled and epoch == 1:
            print(f"[epoch {epoch}] mixup/cutmix: ON "
                f"(alpha={getattr(args,'mixup_alpha',0.0)}, "
                f"cutmix={getattr(args,'cutmix_alpha',0.0)}, "
                f"prob={getattr(args,'mixup_prob',1.0)}, "
                f"mode={getattr(args,'mixup_mode','batch')})")
        if mixup_enabled and epoch == (args.epochs - max(0, mixup_off_k) + 1):
            print(f"[epoch {epoch}] mixup/cutmix: OFF for final {mixup_off_k} epochs")


        use_balanced_now = (balanced_last_k > 0 and epoch > (args.epochs - balanced_last_k) and balanced_loader is not None)
        if use_balanced_now:
            train_loader = balanced_loader
        elif smooth_on and train_loader_smooth is not None:
            train_loader = train_loader_smooth
        else:
            train_loader = (
                train_loader_weighted
                if (args.sampler == "weighted" and train_loader_weighted is not None)
                else train_loader_shuffle
            )

        # ---- build epoch-wise LA & smoothing ----
        tau_eff = tau_at_epoch(epoch, la_start_epoch, la_ramp_epochs, tau_target)

        adj = (tau_eff * log_prior).view(1, -1)
        adj = adj - adj.mean()

        # Label smoothing: when mixup is ON → soft loss; otherwise use configured LS
        label_smoothing = 0.0 if use_mixup_this_epoch else ls_target
        if epoch <= 3 or epoch % 5 == 0:
            print(f"[epoch {epoch}] tau_eff={tau_eff:.3f}, label_smoothing={label_smoothing:.3f}")


        epoch_stats = train_epoch(
            model, train_loader, optimizer, scaler, device, args, epoch,
            mixup_fn, soft_ce, cb_loss, ldam_loss,
            adj, tau_eff, params_to_clip, ema_model, amp_dtype,
            fine_to_coarse_idx, coarse_classes
        )
        train_loss = epoch_stats["loss"]
        train_acc  = epoch_stats["acc"]
        
        
        train_fine_loss = epoch_stats.get("fine_loss")
        train_fine_acc  = epoch_stats.get("fine_acc")
        train_coarse_acc  = epoch_stats.get("coarse_acc")
        train_coarse_loss = epoch_stats.get("coarse_loss")
        train_coarse_loss_w = epoch_stats.get("coarse_loss_w")
        
        if train_coarse_acc is not None:
            print(f"[train] fine_loss {train_fine_loss:.4f} fine_acc {train_fine_acc:.3f} | "
                f"coarse_loss {train_coarse_loss:.4f} coarse_acc {train_coarse_acc:.3f}")
        
        if scheduler is not None:
            scheduler.step()

        # ---- Validation (student) ----
        model.eval()
        val_total = 0
        val_loss_raw_sum = 0.0
        val_loss_adj_sum = 0.0
        val_correct_raw = 0
        val_correct_adj = 0
        all_val_preds, all_val_targets = [], []

        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=amp_dtype, enabled=torch.cuda.is_available()):
            for batch in val_loader:
                if len(batch) == 3:
                    x, y, sid_batch = batch
                else:
                    x, y = batch; sid_batch = None
              
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                
                x = x.to(dtype=next(model.parameters()).dtype)
                
                feats = model.forward_features(x)                     # [B, D_map or tokens]
                pre_logits = model.forward_head(feats, pre_logits=True)  # [B, D]
                logits = model._cosine_head(pre_logits)               # [B, K]
                
                logits_eff = logits - adj.to(dtype=logits.dtype) if tau_eff > 0 else logits
                
                if bool(getattr(args, "use_coarse_aux", 0)):
                    z_coarse = model.coarse_head(pre_logits)          # [B, G]
                    eta = float(getattr(args, "hier_alpha", 0.2)) * min(1.0, epoch / max(1, int(getattr(args,"hier_warmup_epochs",3))))
                    if eta > 0.0:
                        eps = 1e-6
                        q_coarse_val = F.softmax(z_coarse, dim=1).clamp_min(eps).log()       # [B, G]
                        ftc = fine_to_coarse_idx.to(device)
                        if (ftc < 0).any():
                            ftc = ftc.clone(); ftc[ftc < 0] = 0
                            bonus = q_coarse_val[:, ftc]; bonus[:, (fine_to_coarse_idx < 0)] = 0.0
                        else:
                            bonus = q_coarse_val[:, ftc]
                        logits_eff = logits_eff + eta * bonus
                
                
                if args.use_slide_mask:    
                    logits_eff = apply_slide_mask(logits_eff, sid_batch, allowed_bitmask)

                # Effective metrics (the ones you care about)
                loss_eff = F.cross_entropy(logits_eff, y)
                val_loss_adj_sum += loss_eff.item() * y.size(0)
                c_eff, t_ = compute_metrics(logits_eff, y)
                val_correct_adj += c_eff
                val_total += t_
                preds_eff = logits_eff.argmax(1)

                # Raw (optional diagnostics)
                loss_raw = F.cross_entropy(logits, y)
                val_loss_raw_sum += loss_raw.item() * y.size(0)
                c_raw, _ = compute_metrics(logits, y)
                val_correct_raw += c_raw
                
                all_val_preds.append(preds_eff)
                all_val_targets.append(y)

        if all_val_preds:
            all_preds = torch.cat(all_val_preds, dim=0).cpu()
            all_targets = torch.cat(all_val_targets, dim=0).cpu()
        else:
            all_preds = torch.empty(0, dtype=torch.long)
            all_targets = torch.empty(0, dtype=torch.long)

        # Coarse metrics
        coarse_metrics, coarse_cm = compute_coarse_metrics_from_fine(
            all_preds, all_targets, fine_to_coarse_idx, coarse_classes
        )
        print(f"[coarse] macro_recall={coarse_metrics['coarse_macro_recall']:.3f}  "
            f"overall_acc={coarse_metrics['coarse_overall_acc']:.3f}")

        # Optionally save coarse confusions every 10 epochs
        if epoch % 10 == 0 or epoch == args.epochs:
            save_coarse_confusions_from_fine(
                all_targets.numpy(), all_preds.numpy(),
                coarse_classes, fine_to_coarse_idx,
                args.out_dir, epoch=epoch,
                norms=("true", "none", "pred"), print_table_for="true", annotate=True
            )

   
        val_loss_raw = val_loss_raw_sum / max(1, val_total)
        val_loss_adj = val_loss_adj_sum / max(1, val_total)
        val_acc_raw = val_correct_raw / max(1, val_total)
        val_acc_adj = val_correct_adj / max(1, val_total)
        secs = time.time() - t0
        
        

        per_class_metrics = None
        if len(all_preds) > 0:
            per_class_metrics, _, _ = compute_per_class_metrics(all_preds, all_targets, classes)

        macro_f1 = None
        if per_class_metrics:
            macro_f1 = float(per_class_metrics.get("macro_f1", float("nan")))
            if np.isnan(macro_f1):
                macro_f1 = float(
                    f1_score(all_targets.numpy(), all_preds.numpy(), average="macro", zero_division=0)
                )

            print(f"Epoch {epoch:02d}/{args.epochs}  |  "
                  f"train_loss {train_loss:.4f}  train_acc {train_acc:.3f}  "
                  f"val_loss_adj {val_loss_adj:.4f}  val_acc_adj {val_acc_adj:.3f}  "
                  f"(val_loss_raw {val_loss_raw:.4f}  val_acc_raw {val_acc_raw:.3f})  "
                  f"macro_f1 {macro_f1 if macro_f1 is not None else float('nan'):.3f}  "
                  f"| tau_eff {tau_eff:.3f}  ls {label_smoothing:.3f} |  {secs:.1f}s")
            print_per_class_summary(per_class_metrics, classes)
            # Temperature & adj monitor
            if epoch <= 3 or epoch % 5 == 0:
                adj_vals = (-(adj)).squeeze().detach().cpu().numpy()
                mi, ma = int(np.argmin(adj_vals)), int(np.argmax(adj_vals))
                temp = float(F.softplus(getattr(model, "_cosine_head")._s_unconstrained).item())
                print(f"  Cosine temperature: {temp:.2f} | Adjustment extremes: "
                      f"{classes[ma]} ({adj_vals[ma]:+.2f}), {classes[mi]} ({adj_vals[mi]:+.2f})")
            # Optional every 10 epochs
            detailed_prediction_analysis(all_preds, all_targets, classes, epoch)
            if epoch % 10 == 0 or epoch == args.epochs:
                #hard = ["Myeloid cells", "T cells", "Pericytes"]  # hard for fine labels
                #hard = ["Stromal cells", "Epithelial cells", "Endothelial cells"] #hard for coarse labels
                #print_confusion_slice(all_targets.numpy(), all_preds.numpy(), classes, hard, topk=4)
                
                cm_paths = save_confusion_matrices(
                    all_targets.numpy(),
                    all_preds.numpy(),
                    classes,
                    args.out_dir,
                    epoch=epoch,
                    norms=("true", "none", "pred"),      # or ("none","true","pred","all")
                    print_table_for="true",      # prints one table to stdout
                    annotate=True
                )
                
                # Log confusion PNGs to W&B if available
                if wb_run is not None:
                    try:
                        for tag, paths in cm_paths.items():
                            wandb_mod.log(
                                {f"val/confusion_{tag}": wandb_mod.Image(str(paths["png"])), "epoch": epoch},
                                step=epoch
                            )
                    except Exception:
                        pass
                
        else:
            print(f"Epoch {epoch:02d}/{args.epochs}  |  "
                  f"train_loss {train_loss:.4f}  train_acc {train_acc:.3f}  "
                  f"val_loss_adj {val_loss_adj:.4f}  val_acc_adj {val_acc_adj:.3f}  "
                  f"(val_loss_raw {val_loss_raw:.4f}  val_acc_raw {val_acc_raw:.3f})  "
                  f"| tau_eff {tau_eff:.3f}  ls {label_smoothing:.3f} |  {secs:.1f}s")

        # ---- EMA eval (optional) ----
        ema_metrics = None
        if ema_model is not None and ema_eval:
            ema_metrics = evaluate_with_mask(
                ema_model.module, val_loader, device, amp_dtype,
                adj=(adj if tau_eff > 0 else None),
                use_slide_mask=bool(getattr(args, "use_slide_mask", False)),
                allowed_bitmask=allowed_bitmask,
                classes=classes
            )


        # ---- Logging ----
        lr_head = find_group_lr(optimizer, set(head_weight_params)) or 0.0
        lr_lora = find_group_lr(optimizer, set(lora_params) if isinstance(lora_params, list) else set()) or 0.0
        lr_backbone = 0.0
        if unfreeze:
            head_param_set = set(p for p in getattr(model, "_cosine_head").parameters())
            backbone_set = set(p for p in model.parameters() if p.requires_grad and p not in head_param_set)
            lr_backbone = find_group_lr(optimizer, backbone_set) or 0.0

        log_metrics(
            metrics_w, run_id, epoch, train_loss, train_acc, val_loss_adj, val_acc_adj,
            lr_head, lr_lora, secs, per_class_metrics, classes
        )
        metrics_f.flush()
        
        
        base_metrics = {
            "train_loss": train_loss,
            "train_acc":  train_acc,
            "train/fine_loss": train_fine_loss,
            "train/fine_acc":  train_fine_acc,
            "val_loss_eff": val_loss_adj,   # effective = LA (+mask)
            "val_acc_eff":  val_acc_adj,
            "val_loss_raw": val_loss_raw,
            "val_acc_raw":  val_acc_raw,
            "macro_f1":     float(macro_f1 if macro_f1==macro_f1 else 0.0),
            "tau_eff":      tau_eff,
            "label_smoothing": label_smoothing,
            "epoch_secs":   secs,
        }
        
        base_metrics["val_coarse/acc"] = coarse_metrics["coarse_overall_acc"]
        base_metrics["val_coarse/macro_recall"] = coarse_metrics["coarse_macro_recall"]
        
        if train_coarse_acc is not None:
            base_metrics["train_coarse/acc"]   = train_coarse_acc
        if train_coarse_loss is not None:
            base_metrics["train_coarse/loss"]  = train_coarse_loss
        if train_coarse_loss_w is not None:
            base_metrics["train_coarse/loss_w"] = train_coarse_loss_w
            
        lrs = {
            "head": lr_head,
            "head_w": find_group_lr(optimizer, set(head_weight_params)) or 0.0,
            "head_s": find_group_lr(optimizer, set(head_scale_params)) or 0.0,
            "lora": find_group_lr(optimizer, set(lora_params) if isinstance(lora_params, list) else set()) or 0.0,
            "backbone": lr_backbone,
        }

        log_to_wandb(wandb_mod, wb_run, epoch, base_metrics, ema_metrics, lrs)


        # ---- Save best (student vs EMA) ----
        metric_now = (val_acc_adj if save_by == "acc" else (macro_f1 if macro_f1==macro_f1 else 0.0))
        metric_src = "student"
        if ema_metrics is not None:
            ema_now = (ema_metrics["val_eff/acc"] if save_by == "acc" else ema_metrics["macro_f1"])
            if ema_now >= metric_now:
                metric_now = ema_now
                metric_src = "ema"

        if metric_now > best_metric:
            best_metric = metric_now
            to_save = (ema_model.module.state_dict() if (ema_model is not None and metric_src == "ema")
                       else model.state_dict())
            torch.save({
                "model": to_save,
                "args": vars(args),
                "classes": classes,
                "val_acc_eff": (ema_metrics["val_eff/acc"] if (ema_metrics and metric_src=="ema") else val_acc_adj),
                "macro_f1": float((ema_metrics["macro_f1"] if (ema_metrics and metric_src=="ema")
                                else (macro_f1 if macro_f1==macro_f1 else 0.0))),
                "best_metric_name": save_by,
                "best_metric_value": best_metric,
                "ema": bool(ema_model is not None and metric_src=="ema"),
                "ema_present": bool(ema_model is not None),
                "ema_used_for_best": (metric_src == "ema"),
                "ema_decay": float(ema_decay),  
            }, best_path)
            print(f"  ✓ Saved new best ({save_by}={best_metric:.3f}, from={metric_src}) to {best_path}")

        # Early stopping (macro‑F1) — off in your acc‑first runs unless you enable it
        if (not early_stop_off) and (macro_f1 is not None) and (macro_f1==macro_f1):
            if early_stopper.step(macro_f1):
                print(f"\nEarly stopping: no macro F1 improvement for {early_stopper.patience} epochs")
                break
            

    if metrics_f is not None:
        metrics_f.close()
    try:
        if 'wb_run' in locals() and wb_run is not None:
            wb_run.finish()
    except Exception:
        pass

    print(f"\nDone. Best {save_by}={best_metric:.3f}")
    print(f"Checkpoint: {best_path}")
    
    
def train_epoch(
    model,
    train_loader,
    optimizer,
    scaler,
    device,
    args,
    epoch: int,
    mixup_fn,
    soft_ce,
    cb_loss,
    ldam_loss,
    adj: torch.Tensor,
    tau_eff: float,
    params_to_clip,
    ema_model,
    amp_dtype: torch.dtype,
    fine_to_coarse_idx: torch.Tensor = None, 
    coarse_classes: List[str] = None, 
):
    """
    One training epoch with optional coarse auxiliary.
    Returns dict with:
      - total loss/acc
      - fine loss/acc
      - coarse loss/acc (if enabled)
    """
    model.train()

    # running sums
    total_loss_sum = 0.0
    fine_loss_sum  = 0.0
    fine_correct   = 0
    fine_total     = 0

    coarse_loss_sum    = 0.0   # unweighted CE over masked examples
    coarse_loss_w_sum  = 0.0   # weighted contribution (beta warmup applied)
    coarse_correct     = 0
    coarse_total       = 0

    mixup_off_k = int(getattr(args, "mixup_off_epochs", 0))
    use_mixup = bool(mixup_fn is not None and args.loss_type == "ce"
                     and epoch <= (int(args.epochs) - max(0, mixup_off_k)))

    use_coarse_aux = bool(getattr(args, "use_coarse_aux", 0))
    beta  = float(getattr(args, "coarse_loss_weight", 0.3))
    warm  = int(getattr(args, "hier_warmup_epochs", 3))
    eta0  = float(getattr(args, "hier_alpha", 0.2))
    w_coarse = beta * min(1.0, epoch / max(1, warm))
    eta      = eta0 * min(1.0, epoch / max(1, warm))

    for batch in train_loader:
        if len(batch) >= 2:
            x, y = batch[:2]
        else:
            raise RuntimeError("train_loader must yield (x, y, [slide_id])")

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y_hard = y.clone()
        B = y_hard.size(0)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
            # Mixup (soft targets)
            if use_mixup:
                x, y = mixup_fn(x, y)

            # Forward to fine logits and apply LA
            feats = model.forward_features(x) # [B, D_map or tokens]
            pre_logits = model.forward_head(feats, pre_logits=True)  # [B, D]
            logits = model._cosine_head(pre_logits)               # [B, K]
            
            logits_eff = logits - adj.to(dtype=logits.dtype) if tau_eff > 0 else logits

            # ---- Coarse aux: loss + optional soft gating ----
            # (Requires fine_to_coarse_idx and model.coarse_head to exist)
            if use_coarse_aux and hasattr(model, "coarse_head"):
                z_coarse = model.coarse_head(pre_logits)          # [B, G]
                # map fine labels -> coarse labels
                y_coarse = fine_to_coarse_idx[y_hard]        # [B]
                mask = (y_coarse >= 0)

                if mask.any():
                    # a) coarse CE with warm-up weight
                    L_coarse = F.cross_entropy(z_coarse[mask], y_coarse[mask])
                    # we add only the weighted contribution to the total loss
                    # but we log both weighted and unweighted
                    loss_coarse_w = w_coarse * L_coarse
                    # b) soft gating on fine logits (bias toward coarse posterior)
                    if eta > 0.0:
                        eps = 1e-6
                        log_qc = F.softmax(z_coarse, dim=1).clamp_min(eps).log()   # [B, G]
                        ftc = fine_to_coarse_idx
                        if (ftc < 0).any():
                            ftc = ftc.clone(); ftc[ftc < 0] = 0
                            bonus = log_qc[:, ftc]                                  # [B, K]
                            bonus[:, (fine_to_coarse_idx < 0)] = 0.0
                        else:
                            bonus = log_qc[:, ftc]
                        logits_eff = logits_eff + eta * bonus
                else:
                    # no mapped samples in this batch
                    L_coarse = None
                    loss_coarse_w = None
            else:
                L_coarse = None
                loss_coarse_w = None

            # ---- Fine loss (after optional gating) ----
            if args.loss_type == "ce":
                if use_mixup and soft_ce is not None:
                    fine_loss = soft_ce(logits_eff, y)  # soft labels
                else:
                    ls = float(getattr(args, "label_smoothing", 0.0))
                    fine_loss = F.cross_entropy(logits_eff, y, label_smoothing=ls)
                total_loss = fine_loss + (loss_coarse_w if loss_coarse_w is not None else 0.0)
            elif args.loss_type == "cb_focal":
                if cb_loss is None:
                    raise ValueError("CB-Focal requested but cb_loss is None")
                fine_loss = cb_loss(logits_eff, y_hard)
                total_loss = fine_loss  # we don't combine coarse with focal by default
            elif args.loss_type == "ldam_drw":
                if ldam_loss is None:
                    raise ValueError("LDAM-DRW requested but ldam_loss is None")
                fine_loss = ldam_loss(logits_eff, y_hard, epoch=epoch)
                total_loss = fine_loss
            else:
                raise ValueError(f"Unknown loss_type: {args.loss_type}")

        # Backward
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        if ema_model is not None:
            ema_model.update(model)

        # ---- Accumulate metrics ----
        # total & fine
        total_loss_sum += float(total_loss.item()) * B
        fine_loss_sum  += float(fine_loss.item())  * B
        c, t_ = compute_metrics(logits_eff.detach(), y_hard)
        fine_correct += c
        fine_total   += t_

        # coarse
        if use_coarse_aux and L_coarse is not None and mask.any():
            # L_coarse is mean over masked subset
            m = int(mask.sum().item())
            coarse_loss_sum   += float(L_coarse.item())    * m
            coarse_loss_w_sum += float((loss_coarse_w if loss_coarse_w is not None else 0.0).item()) * m
            pred_c = z_coarse.detach().argmax(1)
            coarse_correct += int((pred_c[mask] == y_coarse[mask]).sum().item())
            coarse_total   += m

    # Averages
    out = {
        "loss":            total_loss_sum / max(1, fine_total),
        "acc":             fine_correct   / max(1, fine_total),
        "samples":         fine_total,
        "fine_loss":       fine_loss_sum  / max(1, fine_total),
        "fine_acc":        fine_correct   / max(1, fine_total),
    }
    if use_coarse_aux:
        out.update({
            "coarse_loss":      (coarse_loss_sum   / max(1, coarse_total)) if coarse_total > 0 else None,
            "coarse_loss_w":    (coarse_loss_w_sum / max(1, coarse_total)) if coarse_total > 0 else None,
            "coarse_acc":       (coarse_correct    / max(1, coarse_total)) if coarse_total > 0 else None,
            "coarse_samples":   coarse_total,
        })
    return out



if __name__ == "__main__":
    main()
