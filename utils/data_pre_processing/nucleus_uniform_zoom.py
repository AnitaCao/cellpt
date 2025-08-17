#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import math


def imread_gray(path):
    """Read image as uint8 grayscale."""
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return arr


def resize_with_zoom(img, zoom, is_mask=False):
    """Uniformly scale an image by zoom factor."""
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * zoom)))
    new_h = max(1, int(round(h * zoom)))
    if is_mask:
        interp = cv2.INTER_NEAREST
    else:
        # cubic for upscaling, area for downscaling
        interp = cv2.INTER_CUBIC if (zoom >= 1.0) else cv2.INTER_AREA
    return cv2.resize(img, (new_w, new_h), interpolation=interp)

def feather_composite(img_u8, mask_u8, feather_px=1.0):
    """
    Return (feathered_image_u8, soft_mask_u8). Outside remains black.
    """
    if feather_px <= 0:
        return img_u8, mask_u8
    m = (mask_u8 > 0).astype(np.float32)          # 0..1
    # Gaussian-blur the binary mask to get a soft alpha near the edge
    k = int(2 * math.ceil(3 * feather_px) + 1)    # ~3σ on each side
    alpha = cv2.GaussianBlur(m, (k, k), feather_px)
    alpha = np.clip(alpha, 0.0, 1.0)
    out = (img_u8.astype(np.float32) * alpha).astype(np.uint8)
    soft_mask = (alpha * 255.0).astype(np.uint8)
    return out, soft_mask

def center_crop_or_pad(img, out_size, is_mask=False, pad_value=0):
    """Center-crop to square out_size. Pad with pad_value if needed."""
    h, w = img.shape[:2]
    size = out_size

    # If larger than target, crop
    y0 = max(0, (h - size) // 2)
    x0 = max(0, (w - size) // 2)
    y1 = min(h, y0 + size)
    x1 = min(w, x0 + size)
    cropped = img[y0:y1, x0:x1]

    # If smaller in any dimension, pad
    pad_top = max(0, (size - cropped.shape[0]) // 2)
    pad_bottom = max(0, size - cropped.shape[0] - pad_top)
    pad_left = max(0, (size - cropped.shape[1]) // 2)
    pad_right = max(0, size - cropped.shape[1] - pad_left)

    if pad_top or pad_bottom or pad_left or pad_right:
        border_type = cv2.BORDER_CONSTANT
        value = int(pad_value)
        cropped = cv2.copyMakeBorder(
            cropped, pad_top, pad_bottom, pad_left, pad_right, border_type, value=value
        )
    return cropped


def compute_signal_fraction(mask, thresh=0):
    """Fraction of pixels considered signal. For mask, >0 is foreground."""
    total = mask.size
    if total == 0:
        return 0.0
    # mask is uint8 [0,255]; foreground > 0
    return float((mask > thresh).sum()) / float(total)


def main():
    ap = argparse.ArgumentParser(
        description="Uniform zoom + center-crop nuclei images (and masks) "
                    "to create size-consistent training inputs."
    )
    ap.add_argument("--meta_csv", required=True,
                    help="Path to slide meta CSV (e.g. .../<SLIDE>/meta/nucleus_shapes.csv)")
    ap.add_argument("--zoom", type=float, default=2.0,
                    help="Uniform zoom factor (>1.0 zooms in, <1.0 zooms out).")
    ap.add_argument("--out_size", type=int, default=224,
                    help="Output square size in pixels.")
    ap.add_argument("--suffix", default=None,
                    help="Filename suffix (default: _u{zoom}).")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip saving if the uniform file already exists.")
    ap.add_argument("--feather_px", type=float, default=0.0,
                help="Feather radius in pixels at the mask boundary (0 = off)")
    args = ap.parse_args()

    meta_path = Path(args.meta_csv)
    if not meta_path.is_file():
        raise FileNotFoundError(f"Meta CSV not found: {meta_path}")

    df = pd.read_csv(meta_path)

    # Basic columns required
    needed = {"img_path", "mask_path"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"CSV missing columns: {missing}")

    # Filter to rows we can use
    usable = df[df["img_path"].map(lambda p: Path(str(p)).is_file()) &
                df["mask_path"].map(lambda p: Path(str(p)).is_file())].copy()
    if "skipped_reason" in usable.columns:
        usable = usable[usable["skipped_reason"].fillna("") == ""].copy()

    if len(usable) == 0:
        raise RuntimeError("No usable rows (check paths and skipped_reason).")

    # Figure out slide root (assumes img_path like .../<SLIDE>/img/xxx.png)
    # => slide_root is parent of 'img' dir
    first_img_dir = Path(usable.iloc[0]["img_path"]).parent
    slide_root = first_img_dir.parent

    out_img_dir = slide_root / "img_uniform_3x"
    out_msk_dir = slide_root / "mask_uniform_3x"
    out_meta_dir = slide_root / "meta"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_msk_dir.mkdir(parents=True, exist_ok=True)
    out_meta_dir.mkdir(parents=True, exist_ok=True)

    # New columns
    suffix = args.suffix if args.suffix is not None else f"_u{args.zoom:.2f}"
    df["img_path_uniform"] = ""
    df["mask_path_uniform"] = ""
    df["uniform_zoom"] = np.nan
    df["uniform_out_size"] = np.nan
    df["signal_frac_before"] = np.nan
    df["signal_frac_after"] = np.nan

    processed, skipped = 0, 0
    for idx, row in usable.iterrows():
        try:
            img_path = Path(row["img_path"])
            msk_path = Path(row["mask_path"])

            # Read
            img = imread_gray(img_path)
            msk = imread_gray(msk_path)

            if img.shape != msk.shape:
                raise ValueError(f"Image/Mask shape mismatch: {img.shape} vs {msk.shape} ({img_path.name})")

            # Stats before
            frac_before = compute_signal_fraction(msk)

            # Zoom
            img_z = resize_with_zoom(img, args.zoom, is_mask=False)
            msk_z = resize_with_zoom(msk, args.zoom, is_mask=True)

            # Center crop/pad
            img_o = center_crop_or_pad(img_z, args.out_size, is_mask=False, pad_value=0)
            msk_o = center_crop_or_pad(msk_z, args.out_size, is_mask=True, pad_value=0)
            
            img_o, msk_soft = feather_composite(img_o, msk_o, feather_px=args.feather_px)

            # Stats after
            frac_after = compute_signal_fraction(msk_o)

            # Save
            stem = img_path.stem
            out_img = out_img_dir / f"{stem}{suffix}.png"
            out_msk = out_msk_dir / f"{msk_path.stem}{suffix}.png"

            if not (args.skip_existing and out_img.exists() and out_msk.exists()):
                cv2.imwrite(str(out_img), img_o)
                cv2.imwrite(str(out_msk), msk_o)

            # Record in df
            df.at[idx, "img_path_uniform"] = str(out_img.resolve())
            df.at[idx, "mask_path_uniform"] = str(out_msk.resolve())
            df.at[idx, "uniform_zoom"] = args.zoom
            df.at[idx, "uniform_out_size"] = args.out_size
            df.at[idx, "signal_frac_before"] = frac_before
            df.at[idx, "signal_frac_after"] = frac_after

            processed += 1
            if processed % 1000 == 0:
                print(f"Processed {processed} images...", flush=True)

        except Exception as e:
            skipped += 1
            print(f"Skip idx {idx}: {e}")

    print(f"\nDone: processed={processed}, skipped={skipped}")

    # Write new meta
    out_meta_csv = out_meta_dir / "nucleus_shapes_uniform_3x.csv"
    df.to_csv(out_meta_csv, index=False)
    print(f"New meta written to: {out_meta_csv}")

    # Quick summary
    valid_rows = df["img_path_uniform"].astype(str) != ""
    if valid_rows.any():
        print("\nSummary (uniform):")
        print(f"  Rows with uniform images: {valid_rows.sum()}")
        print(f"  Mean signal fraction before: {df.loc[valid_rows, 'signal_frac_before'].mean():.4f}")
        print(f"  Mean signal fraction after : {df.loc[valid_rows, 'signal_frac_after'].mean():.4f}")


if __name__ == "__main__":
    main()
