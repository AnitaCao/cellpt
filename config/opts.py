# config/opts.py
import argparse
import json
from pathlib import Path

# -----------------------------
# Arg groups
# -----------------------------
def add_common_args(p: argparse.ArgumentParser):
    # dataset / io (not "required" here—validated after config merge)
    p.add_argument("--train_csv", type=str, default=None)
    p.add_argument("--val_csv",   type=str, default=None)
    p.add_argument("--class_map", type=str, default=None)
    p.add_argument("--out_dir",   type=str, default=None)
    p.add_argument("--use_img_uniform", action="store_true")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--use_weighted_sampler", action="store_true",
                    help="Use WeightedRandomSampler for the training DataLoader")
    p.add_argument("--label_col", type=str, default="cell_type")
    p.add_argument("--prior_csv", type=str, default=None,
                   help="Optional CSV to compute class priors for logit adjustment")
    
    p.add_argument("--unfreeze_backbone", action="store_true",
                   help="Unfreeze the whole backbone for full finetuning (skip LoRA)")
    p.add_argument("--lr_backbone", type=float, default=1e-4,
                   help="Backbone LR when unfreeze_backbone is set")


    # model / training
    p.add_argument("--model", type=str, default="vit_base_patch14_dinov2")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    # Explicit base sampler: 'uniform' (shuffle) or 'weighted' (1/n_c).
    p.add_argument("--sampler", choices=["uniform", "weighted"], default="uniform")

    p.add_argument("--cosine", action="store_true")
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    
    # --- Imbalance control (per-class cap) ---
    p.add_argument("--cap_ratio", type=float, default=0.0,
                   help="If > 0: cap each class count in TRAIN to cap_ratio * base_count (see --cap_base). "
                        "Applied after JSON filtering and optional head downsample; 0 disables.")
    p.add_argument("--cap_base", type=str, default="min", choices=["min","p1","p5","p10"],
                   help="Base count for ratio cap: min (default) or low percentile of TRAIN class counts.")

    
    
    # --- Imbalance-aware loss switches ---
    p.add_argument('--loss_type', default='ce',
                   choices=['ce', 'cb_focal', 'ldam_drw'],
                   help="Classification loss: cross-entropy (ce), class-balanced focal (cb_focal), or LDAM with DRW (ldam_drw)")
    p.add_argument('--focal_gamma', type=float, default=1.5)
    p.add_argument('--cb_beta', type=float, default=0.9999,
                   help="CB effective-number beta; 0.9999 is standard for long-tailed data")
    p.add_argument('--ldam_max_m', type=float, default=0.5,
                   help="LDAM max margin")
    p.add_argument('--ldam_s', type=float, default=30.0,
                   help="LDAM scaling factor")
    p.add_argument('--drw_start_epoch', type=int, default=15,
                   help="Start epoch for DRW reweighting (effective-number class weights)")

    

    # Mixup / CutMix (Stage-1 augmentation)
    p.add_argument("--mixup_alpha", type=float, default=0.0,
                   help="Mixup alpha; 0 to disable")
    p.add_argument("--cutmix_alpha", type=float, default=0.0,
                   help="CutMix alpha; 0 to disable")
    p.add_argument("--mixup_prob", type=float, default=1.0,
                   help="Probability to apply mixup/cutmix per batch")
    p.add_argument("--mixup_switch_prob", type=float, default=0.0,
                   help="Probability to switch between mixup/cutmix in a batch")
    p.add_argument("--mixup_mode", type=str, default="batch",
                   help="Mixup application mode (batch, pair, elem)")
    p.add_argument("--mixup_off_epochs", type=int, default=10,
                   help="Turn OFF mixup/cutmix for the last K epochs")

    # config plumbing
    p.add_argument("--config", type=str, help="Path to JSON config with defaults")
    p.add_argument("--save_metric", type=str, default="acc",
                   help="Metric to use for model selection")
    p.add_argument("--save_args", action="store_true",
                   help="Write merged args to out_dir/args.resolved.json")
    return p


def add_lora_args(p: argparse.ArgumentParser):
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_lora", type=float, default=3e-4)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument("--lora_blocks", type=int, default=6)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--logit_tau", type=float, default=0.1)
    
    p.add_argument("--la_start_epoch", type=int, default=3,
                   help="Epoch to start logit adjustment")  
    p.add_argument("--la_ramp_epochs", type=int, default=3,
                   help="Epochs to ramp up logit adjustment")
    return p


# -----------------------------
# Parser builder + parse
# -----------------------------
def _build_parser(build_fn):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_common_args(parser)
    parser = build_fn(parser)
    return parser


def parse_args(build_fn):
    """
    build_fn: function(parser) -> parser that adds trainer-specific args.
    Precedence: CLI > JSON config > defaults.
    """
    # Pass 1: only grab --config without tripping 'required' checks
    cfg_only = argparse.ArgumentParser(add_help=False)
    cfg_only.add_argument("--config", type=str)
    cfg_args, remaining = cfg_only.parse_known_args()

    # Build the full parser
    parser = _build_parser(build_fn)

    # If a config is provided, load it as defaults
    if cfg_args.config:
        cfg_path = Path(cfg_args.config)
        if not cfg_path.exists():
            raise SystemExit(f"--config not found: {cfg_path}")
        with cfg_path.open("r") as f:
            cfg = json.load(f)
        # Only apply keys that exist as known args
        known_dests = {a.dest for a in parser._actions}
        parser.set_defaults(**{k: v for k, v in cfg.items() if k in known_dests})

    # Pass 2: parse remaining CLI; these override JSON/defaults
    args = parser.parse_args(remaining)

    # Validate required after merge
    required = ["train_csv", "val_csv", "class_map", "out_dir"]
    missing = [k for k in required if not getattr(args, k)]
    if missing:
        parser.error(
            "Missing required args (supply via --config or CLI): " + ", ".join(missing)
        )

    # Optionally save the resolved args
    if args.save_args:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        Path(out, "args.resolved.json").write_text(json.dumps(vars(args), indent=2))

    return args
