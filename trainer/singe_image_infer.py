#!/usr/bin/env python3
"""
Single-sample classifier for CellPT checkpoints.
Supports two-view inference with two image paths.
Mirrors eval forward path used in training code.
"""
import argparse, json, time, sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

FILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FILE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Project imports
from trainer.main_trainer_refactor_v3 import build_backbone_and_heads, forward_batch

try:
    import torchvision.transforms as T
except Exception as e:
    raise RuntimeError("torchvision is required for image transforms") from e


def parse_args():
    p = argparse.ArgumentParser("CellPT single inference")
    p.add_argument("--ckpt", type=str, default="/hpc/group/jilab/rz179/cellpt/experiments/multiview_baselines_v3/mv_2p5x_10x_scale_entropyFlat/swin_base_patch4_window7_224_bs36_lrh0.001_20251014-100745.pt", 
                   help="Path to .pt checkpoint saved by training")
    p.add_argument("--img_view1", type=str, default="/hpc/group/jilab/boxuan/data/Xenium_V1_hLiver_cancer/all/Xenium_V1_hLiver_cancer/img/raw/Xenium_V1_hLiver_cancer_n94898.png",
                   help="Path to view 1 image, for example 2.5x")
    p.add_argument("--img_view2", type=str, default="/hpc/group/jilab/boxuan/data_10.0context/Xenium_V1_hLiver_cancer/all/Xenium_V1_hLiver_cancer/img/raw/Xenium_V1_hLiver_cancer_n94898.png", 
                   help="Path to view 2 image, for example 10x")
    p.add_argument("--label_col", default=None, help="Optional. Overrides label col if needed")
    p.add_argument("--apply_la", action="store_true", help="Apply logit adjustment at inference")
    p.add_argument("--tau", type=float, default=None, help="LA temperature. Defaults to value stored in ckpt args")
    p.add_argument("--prior_csv", type=str, default=None, help="CSV with priors used for LA. Defaults to ckpt args prior_csv if present")
    p.add_argument("--topk", type=int, default=5, help="Top K to display")
    p.add_argument("--save_json", type=str, default="/hpc/group/jilab/rz179/cellpt/experiments/signle_view_infer", help="Optional output JSON path for predictions")
    return p.parse_args()


def _safe_load(ckpt_path: str):
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"[infer] weights_only=True failed ({e}); falling back to weights_only=False for this trusted file.")
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def _build_transform(img_size: int, mean, std):
    mean = list(mean)
    std = list(std)
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def _load_image(path: str, tfm):
    img = Image.open(path).convert("RGB")
    return tfm(img)


def _prepare_input(v1, v2, multi_view: bool):
    if multi_view:
        # x shape: [B=1, V=2, C, H, W]
        x = torch.stack([v1, v2], dim=0).unsqueeze(0)
    else:
        # x shape: [B=1, C, H, W]
        x = v1.unsqueeze(0)
    return x


def main():
    args = parse_args()

    ckpt = _safe_load(args.ckpt)
    classes = ckpt["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Recover training args
    from types import SimpleNamespace
    ckpt_args = SimpleNamespace(**ckpt["args"]) if isinstance(ckpt["args"], dict) else ckpt["args"]

    # Build model exactly as in training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    # Ensure required fields
    if not hasattr(ckpt_args, "img_size"):
        setattr(ckpt_args, "img_size", 224)
    if not hasattr(ckpt_args, "multi_view"):
        setattr(ckpt_args, "multi_view", 1 if args.img_view2 else 0)

    model, mean, std, _, _ = build_backbone_and_heads(ckpt_args, classes, num_classes=len(classes), device=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # Transforms
    tfm = _build_transform(int(ckpt_args.img_size), mean, std)

    # Load images
    v1 = _load_image(args.img_view1, tfm)
    v2 = None
    if int(getattr(ckpt_args, "multi_view", 0)) == 1:
        if not args.img_view2:
            raise ValueError("Model expects two views. Provide --img_view2.")
        v2 = _load_image(args.img_view2, tfm)

    x = _prepare_input(v1, v2, multi_view=(int(getattr(ckpt_args, "multi_view", 0)) == 1))
    x = x.to(device)

    # Forward
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
        pre, _, aux = forward_batch(model, x, epoch=0, args=ckpt_args, amp_dtype=amp_dtype)
        logits = aux["fused_logits"] if (aux is not None and "fused_logits" in aux) else model._cosine_head(pre)

        probs_raw = F.softmax(logits, dim=1)[0]

    # Top K
    topk = max(1, int(args.topk))
    pr_raw, idx_raw = torch.topk(probs_raw, k=topk)

    def _fmt(idx_t, pr_t):
        return [{"class": classes[int(i)], "prob": float(p)} for p, i in zip(pr_t.tolist(), idx_t.tolist())]

    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ckpt": str(args.ckpt),
        "multi_view": int(getattr(ckpt_args, "multi_view", 0)) == 1,
        "view1": str(args.img_view1),
        "view2": str(args.img_view2) if args.img_view2 else None,
        "topk": topk,
        "top_raw": _fmt(idx_raw, pr_raw),
    }

    # Print table
    print("\nTop raw predictions:")
    for r in result["top_raw"]:
        print(f"  {r['class']:<30s}  p={r['prob']:.4f}")


    if args.save_json:
        outp = Path(args.save_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote {outp}")


if __name__ == "__main__":
    main()
