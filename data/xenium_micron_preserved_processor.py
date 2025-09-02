#!/usr/bin/env python3
import os, csv, argparse, sys
from pathlib import Path
import numpy as np
import tifffile, zarr, cv2
from PIL import Image
import math
import csv


# ---------- Xenium cell_id helpers ----------
# 8 hex nibbles -> letters a..p mapping per 10x docs
_HEX_TO_AP = str.maketrans("0123456789abcdef", "abcdefghijklmnop")

def encode_cell_id(prefix_uint32, suffix_uint32):
    """Make Xenium string cell_id like 'ffkpbaba-1' from (prefix, suffix)."""
    h = f"{int(prefix_uint32):08x}"              # 8-digit lowercase hex
    return h.translate(_HEX_TO_AP) + f"-{int(suffix_uint32)}"

"""
def load_cell_id_table(zarr_path):
    #Return np.ndarray shape (N,2) uint32 with [prefix, suffix] per cell.
    p = Path(zarr_path)
    store = zarr.ZipStore(str(p), mode="r") if p.suffix == ".zip" else zarr.DirectoryStore(str(p))
    g = zarr.open(store, mode="r")
    return np.asarray(g["cell_id"])
"""

def load_cell_id_table(zarr_path):
    """
    Return an np.ndarray of shape (N, 2) uint32 with [prefix, suffix] per cell.
    Handles 2-D cell_id ([N,2]), 1-D cell_id ([N]), structured dtypes, or
    separate cell_id_prefix/dataset_suffix arrays (with sensible defaults).
    """
    p = Path(zarr_path)
    store = zarr.ZipStore(str(p), mode="r") if p.suffix == ".zip" else zarr.DirectoryStore(str(p))
    g = zarr.open(store, mode="r")

    def has(group, key):
        return (key in group)

    # Prefer cell_id if present (top-level or under "cells")
    arr = None
    if has(g, "cell_id"):
        arr = g["cell_id"]
    elif has(g, "cells") and has(g["cells"], "cell_id"):
        arr = g["cells"]["cell_id"]

    prefixes = None
    suffixes = None

    if arr is not None:
        a = np.asarray(arr)
        # Structured dtype: take first two fields as prefix/suffix if available
        if a.dtype.fields:
            names = list(a.dtype.fields.keys())
            prefixes = np.asarray(a[names[0]], dtype=np.uint32).reshape(-1)
            if len(names) > 1:
                suffixes = np.asarray(a[names[1]], dtype=np.uint32).reshape(-1)
        elif a.ndim == 2:
            prefixes = a[:, 0].astype(np.uint32)
            suffixes = (a[:, 1].astype(np.uint32)
                        if a.shape[1] >= 2
                        else np.ones_like(prefixes, dtype=np.uint32))
        elif a.ndim == 1:
            prefixes = a.astype(np.uint32)
            suffixes = np.ones_like(prefixes, dtype=np.uint32)
        else:
            raise RuntimeError(f"Unsupported cell_id array shape: {a.shape}")

    # Fallback: look for separate prefix/suffix arrays
    if prefixes is None:
        pref = None
        if has(g, "cell_id_prefix"):
            pref = g["cell_id_prefix"]
        elif has(g, "cells") and has(g["cells"], "cell_id_prefix"):
            pref = g["cells"]["cell_id_prefix"]
        if pref is None:
            raise RuntimeError("Could not find cell_id or cell_id_prefix in zarr.")
        prefixes = np.asarray(pref, dtype=np.uint32).reshape(-1)

    if suffixes is None:
        ds = None
        if has(g, "dataset_suffix"):
            ds = g["dataset_suffix"]
        elif has(g, "cells") and has(g["cells"], "dataset_suffix"):
            ds = g["cells"]["dataset_suffix"]
        if ds is None:
            suffixes = np.ones_like(prefixes, dtype=np.uint32)
        else:
            ds = np.asarray(ds)
            if ds.ndim == 0:
                suffixes = np.full_like(prefixes, int(ds), dtype=np.uint32)
            elif ds.ndim == 1 and ds.size in (1, prefixes.size):
                suffixes = (np.full_like(prefixes, int(ds[0]), dtype=np.uint32)
                            if ds.size == 1 else ds.astype(np.uint32))
            else:
                suffixes = np.ones_like(prefixes, dtype=np.uint32)

    return np.stack([prefixes.astype(np.uint32), suffixes.astype(np.uint32)], axis=1)

def load_celltype_map(csv_path):
    """Read a CSV with columns cellid (or cell_id) and celltype -> dict."""
    m = {}
    p = Path(csv_path) if csv_path else None
    if not p or not p.exists():
        print(f"Cell type CSV not found: {csv_path}")
        raise RuntimeError(f"Cell type CSV not found: {csv_path}")
    with p.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        if not rdr.fieldnames:
            return m
        cols = {c.lower(): c for c in rdr.fieldnames}
        k_id = cols.get("cellid") or cols.get("cell_id")
        k_ty = cols.get("celltype")
        if not k_id or not k_ty:
            return m
        for row in rdr:
            cid = (row[k_id] or "").strip()
            cty = (row[k_ty] or "").strip()
            if cid:
                m[cid] = cty
    return m


# ---------- mpp helpers ----------
def get_um_per_px_from_zarr(zarr_path):
    """
    Return (um_per_px_x, um_per_px_y) using masks/homogeneous_transform.
    Handles rotation/shear by taking vector norms of columns.
    """
    p = Path(zarr_path)
    store = zarr.ZipStore(str(p), mode="r") if p.suffix == ".zip" else zarr.DirectoryStore(str(p))
    g = zarr.open(store, mode="r")
    T = np.asarray(g["masks"]["homogeneous_transform"])  # microns -> pixels, 4x4
    px_per_um_x = np.linalg.norm(T[:2, 0])  # column 0 XY norm
    px_per_um_y = np.linalg.norm(T[:2, 1])  # column 1 XY norm
    um_per_px_x = 1.0 / px_per_um_x
    um_per_px_y = 1.0 / px_per_um_y
    print(f"Microns per pixel: X={um_per_px_x:.6f}, Y={um_per_px_y:.6f}")
    return float(um_per_px_x), float(um_per_px_y)

# ---------- I/O helpers ----------
def open_ome_as_zarr(ome_path, series=0, level=0):
    tf = tifffile.TiffFile(str(ome_path))
    store = tf.aszarr(series=series, level=level)   # lazy
    z = zarr.open(store, mode="r")                  # (C,Y,X) or (Y,X)
    return tf, z

def open_nucseg_zarr(zarr_path, channel=0):
    p = Path(zarr_path)
    store = zarr.ZipStore(str(p), mode="r") if p.suffix == ".zip" else zarr.DirectoryStore(str(p))
    g = zarr.open(store, mode="r")
    masks = g["masks"]
    key = str(channel)  # "0" nucleus, "1" cell
    if key not in masks:
        raise RuntimeError(f"masks['{key}'] not found. Available: {list(masks.keys())}")
    mask = masks[key]
    print(f"Using masks['{key}'] shape={mask.shape} dtype={mask.dtype}")
    return mask

def auto_pick_ome(ome_or_dir, zarr_path):
    p = Path(ome_or_dir)
    if p.is_file():
        print(f"Using specified OME file: {p}")
        sys.stdout.flush()
        return str(p)
    lbl = open_nucseg_zarr(zarr_path, channel=0)
    H, W = lbl.shape
    print(f"Looking for OME file matching label dimensions: {H} x {W}")
    cands = sorted((p / "morphology_focus").glob("morphology_focus_*.ome.tif"))
    if not cands and (p / "morphology_focus.ome.tif").exists():
        cands = [p / "morphology_focus.ome.tif"]
    print(f"Found {len(cands)} candidate OME files")
    for cand in cands:
        print(f"Checking {cand}...")
        try:
            with tifffile.TiffFile(str(cand)) as tf:
                z = zarr.open(tf.aszarr(series=0, level=0), mode="r")
                Hy, Wx = (z.shape[1], z.shape[2]) if len(z.shape) == 3 else (z.shape[0], z.shape[1])
                print(f"  OME dimensions: {Hy} x {Wx}")
                if (Hy, Wx) == (H, W):
                    print("  ✓ Match found!")
                    return str(cand)
                else:
                    print("  ✗ No match")
        except Exception as e:
            print(f"  Error reading {cand}: {e}")
    raise RuntimeError(f"No OME tile matches label shape {H}x{W} in {p}")

# ---------- image helpers ----------
def percentile_norm(img, p1=1.0, p99=99.0):
    lo, hi = np.percentile(img, p1), np.percentile(img, p99)
    hi = max(hi, lo + 1e-6)
    img = np.clip(img, lo, hi)
    return ((img - lo) / (hi - lo) * 255.0).astype(np.uint8)

def tile_ranges(H, W, tile):
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            yield y0, min(H, y0+tile), x0, min(W, x0+tile)

def build_nucleus_index(lbl_z, tile=2048, min_area=20):
    H, W = lbl_z.shape
    print(f"Image size: {H} x {W} pixels")
    sys.stdout.flush()
    max_id = 0
    for y0,y1,x0,x1 in tile_ranges(H,W,tile):
        c = np.asarray(lbl_z[y0:y1, x0:x1])
        if c.size: 
            max_id = max(max_id, int(c.max()))
    npx   = np.zeros(max_id+1, np.int64)
    sum_x = np.zeros(max_id+1, np.float64)
    sum_y = np.zeros(max_id+1, np.float64)
    min_x = np.full(max_id+1, np.inf)
    min_y = np.full(max_id+1, np.inf)
    max_x = np.full(max_id+1, -np.inf)
    max_y = np.full(max_id+1, -np.inf)
    for y0,y1,x0,x1 in tile_ranges(H,W,tile):
        c = np.asarray(lbl_z[y0:y1, x0:x1])
        if not c.size: continue
        ids = np.unique(c); ids = ids[ids!=0]
        for cid in ids:
            m = (c==cid)
            if not m.any(): continue
            ys, xs = np.nonzero(m)
            n = ys.size
            xs_g, ys_g = xs + x0, ys + y0
            npx[cid]   += n
            sum_x[cid] += xs_g.sum()
            sum_y[cid] += ys_g.sum()
            mnx, mxx = xs_g.min(), xs_g.max()
            mny, mxy = ys_g.min(), ys_g.max()
            if mnx < min_x[cid]: min_x[cid] = mnx
            if mny < min_y[cid]: min_y[cid] = mny
            if mxx > max_x[cid]: max_x[cid] = mxx
            if mxy > max_y[cid]: max_y[cid] = mxy
    nuclei=[]
    for cid in range(1, max_id+1):
        if npx[cid] < min_area or not np.isfinite(min_x[cid]): 
            continue
        cx = int(round(sum_x[cid]/npx[cid]))
        cy = int(round(sum_y[cid]/npx[cid]))
        x0, y0 = int(min_x[cid]), int(min_y[cid])
        x1, y1 = int(max_x[cid]), int(max_y[cid])
        nuclei.append({
            "cid": cid, 
            "area_px": int(npx[cid]),
            "centroid": (cx,cy), 
            "bbox": (x0, y0, x1-x0+1, y1-y0+1)
        })
    return nuclei, (H,W)

def read_window_with_padding(zimg, c, x0, y0, side):
    if len(zimg.shape)==3:
        _, H, W = zimg.shape
        crop = np.asarray(zimg[c, max(0,y0):min(H,y0+side), max(0,x0):min(W,x0+side)])
    else:
        H, W = zimg.shape
        crop = np.asarray(zimg[max(0,y0):min(H,y0+side), max(0,x0):min(W,x0+side)])
    pt, pl = max(0,-y0), max(0,-x0)
    pb, pr = max(0,(y0+side)-H), max(0,(x0+side)-W)
    if pt or pl or pb or pr:
        crop = cv2.copyMakeBorder(crop, pt,pb,pl,pr, cv2.BORDER_CONSTANT, value=0)
    return crop

def slice_label_with_padding(lbl_z, x0, y0, side):
    H, W = lbl_z.shape
    crop = np.asarray(lbl_z[max(0,y0):min(H,y0+side), max(0,x0):min(W,x0+side)])
    pt, pl = max(0,-y0), max(0,-x0)
    pb, pr = max(0,(y0+side)-H), max(0,(x0+side)-W)
    if pt or pl or pb or pr:
        crop = cv2.copyMakeBorder(crop.astype(np.int32), pt,pb,pl,pr, cv2.BORDER_CONSTANT, value=0)
    return crop


# ---------- micron calibration + shape extraction ----------
def resample_to_target_mpp(win_img_u8, win_lbl_ids, um_per_px_x, um_per_px_y, target_mpp):
    """
    Calibrate native window to a shared target mpp using separate x and y scales.
    Masks use nearest. Image uses area for downsampling, linear for upsampling.
    """
    scale_x = um_per_px_x / target_mpp
    scale_y = um_per_px_y / target_mpp
    if abs(scale_x - 1.0) < 1e-6 and abs(scale_y - 1.0) < 1e-6:
        return win_img_u8, win_lbl_ids
    H, W = win_img_u8.shape
    newW = max(1, int(round(W * scale_x)))
    newH = max(1, int(round(H * scale_y)))
    interp_img = cv2.INTER_AREA if (newW < W or newH < H) else cv2.INTER_LINEAR
    img_rs = cv2.resize(win_img_u8, (newW, newH), interpolation=interp_img)
    ids_rs = cv2.resize(win_lbl_ids.astype(np.int32), (newW, newH), interpolation=cv2.INTER_NEAREST)
    return img_rs, ids_rs

def extract_nucleus_shape(win_img_u8, win_lbl_ids, cid, out_size=224, jitter=3, bg_type='median'):
    """
    Extract exact nucleus shape and place on clean background - NO coverage filtering
    
    Args:
        win_img_u8: Calibrated image window
        win_lbl_ids: Calibrated label window
        cid: Nucleus ID
        out_size: Output canvas size
        jitter: Random position jitter in pixels
        bg_type: Background type ('median', 'black', 'noise')
        
    Returns:
        tuple: (final_image, final_mask, nucleus_info, skip_reason) or None
    """
    m = (win_lbl_ids == cid)
    if not m.any():
        return None, "nucleus_not_found"
    
    # Extract exact nucleus bounding box
    ys, xs = np.nonzero(m)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    sub_img = win_img_u8[y0:y1+1, x0:x1+1]
    sub_msk = m[y0:y1+1, x0:x1+1].astype(np.uint8) * 255
    H, W = sub_img.shape
    
    # Size validation (replaces coverage filtering)
    max_size = int(out_size * 0.85)  # Allow up to 85% of canvas
    if H > max_size or W > max_size:
        return None, "too_large_for_canvas"
    
    min_size = 3  # Minimum visible size
    if H < min_size or W < min_size:
        return None, "too_small_to_process"

    # Create background
    if bg_type == 'black':
        bg_value = 0
    elif bg_type == 'noise':
        nucleus_mean = np.mean(sub_img[sub_msk > 0]) if np.any(sub_msk > 0) else 50
        bg_base = nucleus_mean * 0.1
        background = np.random.normal(bg_base, 5, (out_size, out_size))
        background = np.clip(background, 0, 255).astype(np.uint8)
    else:  # 'median' - default
        bg_value = int(np.median(win_img_u8))
    
    if bg_type != 'noise':
        background = np.full((out_size, out_size), bg_value, dtype=np.uint8)
    
    # Create mask canvas
    mask_canvas = np.zeros((out_size, out_size), dtype=np.uint8)

    # Calculate placement with optional jitter
    center_y, center_x = out_size // 2, out_size // 2
    
    if jitter > 0:
        max_jitter_y = min(jitter, (out_size - H) // 2)
        max_jitter_x = min(jitter, (out_size - W) // 2)
        jitter_y = np.random.randint(-max_jitter_y, max_jitter_y + 1) if max_jitter_y > 0 else 0
        jitter_x = np.random.randint(-max_jitter_x, max_jitter_x + 1) if max_jitter_x > 0 else 0
    else:
        jitter_y, jitter_x = 0, 0
    
    # Final placement coordinates
    start_y = center_y - H // 2 + jitter_y
    start_x = center_x - W // 2 + jitter_x
    
    # Ensure nucleus stays within bounds
    start_y = max(0, min(start_y, out_size - H))
    start_x = max(0, min(start_x, out_size - W))
    
    # Place nucleus with exact shape
    nucleus_region = background[start_y:start_y+H, start_x:start_x+W]
    mask_region = sub_msk > 0
    
    background[start_y:start_y+H, start_x:start_x+W] = np.where(
        mask_region, sub_img, nucleus_region
    )
    mask_canvas[start_y:start_y+H, start_x:start_x+W] = sub_msk
    
    nucleus_info = {
        'final_width_px': W,
        'final_height_px': H,
        'placed_at_x': start_x,
        'placed_at_y': start_y,
        'jitter_x': jitter_x,
        'jitter_y': jitter_y,
        'bg_type': bg_type
    }
    
    return background, mask_canvas, nucleus_info, "success"

# ---------- main pipeline ----------
def process_nucleus_shapes_mpp(slide_id, ome_or_dir, zarr_path, out_root,
                              channel_index=0, out_size=224,
                              target_mpp=0.50,
                              # Biological size filters
                              # Processing parameters
                              bg_type='median', jitter=3,
                              pad_px=64, tile=2048, min_area=20, 
                              print_every=1000, celltype_csv=""):
    

    print("Starting nucleus shape extraction with MPP calibration...")
    print(f"Target MPP: {target_mpp} μm/pixel")
    print(f"Background type: {bg_type}")


    out_root = Path(out_root)
    img_dir = out_root / "img"
    msk_dir = out_root / "mask"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    meta = out_root / "meta" / "nucleus_shapes.csv"
    meta.parent.mkdir(parents=True, exist_ok=True)
    new = not meta.exists()
    cols = [
        "slide_id", "nucleus_id", "img_path", "mask_path",
        "orig_bbox_w_px", "orig_bbox_h_px", "orig_area_px",
        "biological_diameter_um", "biological_area_um2", "biological_width_um", "biological_height_um",
        "aspect_ratio", "final_width_px", "final_height_px",
        "placed_at_x", "placed_at_y", "jitter_x", "jitter_y",
        "um_per_px_x", "um_per_px_y", "target_mpp", "bg_type",
        "cell_id", "cell_type",
        "skipped_reason"
    ]
    f = open(meta, "a", newline="")
    wr = csv.DictWriter(f, fieldnames=cols)
    if new: 
        wr.writeheader()

    ome_path = auto_pick_ome(ome_or_dir, zarr_path)
    print(f"Using OME file: {ome_path}")
    sys.stdout.flush()

    # Open data
    tf, zimg = open_ome_as_zarr(ome_path)
    lbl_z = open_nucseg_zarr(zarr_path, channel=0)

    # Dimension check
    H, W = lbl_z.shape
    zshape = zimg.shape
    Hy, Wx = (zshape[1], zshape[2]) if len(zshape)==3 else (zshape[0], zshape[1])
    if (Hy, Wx) != (H, W):
        tf.close()
        f.close()
        raise RuntimeError(f"Shape mismatch OME {Hy,Wx} vs Label {H,W}")

    # Get pixel spacing
    um_per_px_x, um_per_px_y = get_um_per_px_from_zarr(zarr_path)

    print("Building nucleus index...")
    nuclei, _ = build_nucleus_index(lbl_z, tile=tile, min_area=min_area)
    
  
    #load cell_id table and cell_type mapping
    try:
        cellid_tbl = load_cell_id_table(zarr_path)
        print(f"Loaded cell_id table: {cellid_tbl.shape[0]} cells")
    except Exception as e:
        print(f"Error loading cell_id table: {e}")
        cellid_tbl = None
    
    ct_map = load_celltype_map(celltype_csv)
    if ct_map:
        print(f"Loaded cell type map: {len(ct_map)} types")
  
  
    n = len(nuclei)
    kept = 0
    skip_stats = {
        'nucleus_not_found': 0,
        'too_large_for_canvas': 0,
        'too_small_to_process': 0,
        'invalid_shape': 0,
        'processing_error': 0
    }
    
    print(f"Processing {n} nuclei...")
    sys.stdout.flush()

    for k, info in enumerate(nuclei, 1):
        cid = info["cid"]
        (cx, cy) = info["centroid"]
        (bx, by, bw, bh) = info["bbox"]
        area_px = info["area_px"]
        
        width_um  = bw * um_per_px_x
        height_um = bh * um_per_px_y
        area_um2  = area_px * (um_per_px_x * um_per_px_y)
        diameter_um = 2.0 * math.sqrt(area_um2 / math.pi) if area_um2 > 0 else 0.0
        aspect_ratio = (max(width_um, height_um) / max(1e-9, min(width_um, height_um)))
        
        
         # Map label id -> Xenium cell_id string (and cell_type)
        cell_id_str = ""
        cell_type_str = ""
        if cellid_tbl is not None:
            idx = int(cid) - 1  # labels are 1-based
            if 0 <= idx < cellid_tbl.shape[0]:
                pref, suff = cellid_tbl[idx, 0], cellid_tbl[idx, 1]
                cell_id_str = encode_cell_id(pref, suff)  # e.g., 'ffkpbaba-1'
                if ct_map:
                    # Try in order: exact string id, numeric prefix, numeric label id
                    cell_type_str = (ct_map.get(cell_id_str)
                                     or ct_map.get(str(int(pref)))
                                     or ct_map.get(str(int(cid)))
                                     or "")
        
        try:
            # Extract window around nucleus
            side = max(bw, bh) + 2 * pad_px
            x0 = max(0, min(cx - side//2, W - side))
            y0 = max(0, min(cy - side//2, H - side))

            # Read native window
            win = read_window_with_padding(zimg, channel_index, x0, y0, side)
            win_u8 = percentile_norm(win)
            crop_ids = slice_label_with_padding(lbl_z, x0, y0, side)

            # Calibrate window to target MPP
            win_u8_rs, crop_ids_rs = resample_to_target_mpp(
                win_u8, crop_ids, um_per_px_x, um_per_px_y, target_mpp
            )

            # Extract nucleus shape
            result = extract_nucleus_shape(
                win_u8_rs, crop_ids_rs, cid, out_size, jitter, bg_type
            )
            
            if result[0] is None:
                # Extraction failed
                skip_reason = result[1]
                skip_stats[skip_reason] = skip_stats.get(skip_reason, 0) + 1
                wr.writerow({
                    "slide_id": slide_id, "nucleus_id": int(cid),
                    "img_path": "", "mask_path": "",
                    "orig_bbox_w_px": int(bw), "orig_bbox_h_px": int(bh),
                    "orig_area_px": int(area_px),
                    "biological_diameter_um": float(diameter_um),
                    "biological_area_um2": float(area_um2),
                    "biological_width_um": float(width_um),
                    "biological_height_um": float(height_um),
                    "aspect_ratio": float(aspect_ratio),
                    "final_width_px": 0, "final_height_px": 0,
                    "placed_at_x": 0, "placed_at_y": 0, "jitter_x": 0, "jitter_y": 0,
                    "um_per_px_x": um_per_px_x, "um_per_px_y": um_per_px_y,
                    "target_mpp": float(target_mpp), "bg_type": bg_type,
                    "cell_id": cell_id_str, "cell_type": cell_type_str,
                    "skipped_reason": skip_reason
                })
                continue

            final_img, final_mask, nucleus_info, _ = result

            # Save files
            img_name = f"{slide_id}_nucleus_{cid}.png"
            msk_name = f"{slide_id}_nucleus_{cid}_mask.png"
            
            cv2.imwrite(str(img_dir / img_name), final_img)
            Image.fromarray(final_mask).save(msk_dir / msk_name)

            # Record successful extraction
            wr.writerow({
                "slide_id": slide_id,
                "nucleus_id": int(cid),
                "img_path": str((img_dir / img_name).resolve()),
                "mask_path": str((msk_dir / msk_name).resolve()),
                "orig_bbox_w_px": int(bw),
                "orig_bbox_h_px": int(bh),
                "orig_area_px": int(area_px),
                "biological_diameter_um": float(diameter_um),
                "biological_area_um2": float(area_um2),
                "biological_width_um": float(width_um),
                "biological_height_um": float(height_um),
                "aspect_ratio": float(aspect_ratio),
                "final_width_px": nucleus_info['final_width_px'],
                "final_height_px": nucleus_info['final_height_px'],
                "placed_at_x": nucleus_info['placed_at_x'],
                "placed_at_y": nucleus_info['placed_at_y'],
                "jitter_x": nucleus_info['jitter_x'],
                "jitter_y": nucleus_info['jitter_y'],
                "um_per_px_x": um_per_px_x,
                "um_per_px_y": um_per_px_y,
                "target_mpp": float(target_mpp),
                "bg_type": bg_type,
                "cell_id": cell_id_str,
                "cell_type": cell_type_str,
                "skipped_reason": ""
            })
            
            kept += 1
            if k % print_every == 0:
                print(f"{slide_id}: {k}/{n} processed, {kept} kept")

        except Exception as e:
            print(f"Error processing nucleus {cid}: {e}")
            skip_stats['processing_error'] += 1
            wr.writerow({
                "slide_id": slide_id, "nucleus_id": int(cid),
                "img_path": "", "mask_path": "",
                "orig_bbox_w_px": int(bw), "orig_bbox_h_px": int(bh),
                "orig_area_px": int(area_px),
                "biological_diameter_um": float(diameter_um),
                "biological_area_um2": float(area_um2),
                "biological_width_um": float(width_um),
                "biological_height_um": float(height_um),
                "aspect_ratio": float(aspect_ratio),
                "final_width_px": 0, "final_height_px": 0,
                "placed_at_x": 0, "placed_at_y": 0, "jitter_x": 0, "jitter_y": 0,
                "um_per_px_x": um_per_px_x, "um_per_px_y": um_per_px_y,
                "target_mpp": float(target_mpp), "bg_type": bg_type,
                "skipped_reason": f"processing_error_{str(e)[:50]}"
            })
            continue

    # Print final statistics
    print(f"\n{slide_id}: Processing complete!")
    print(f"Successfully extracted: {kept}/{n} nucleus shapes")
    print(f"Skip reasons:")
    for reason, count in skip_stats.items():
        if count > 0:
            print(f"  {reason}: {count}")

    f.close()
    tf.close()
    return kept

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Extract nucleus shapes with biological scale preservation - no coverage filtering")
    ap.add_argument("--slide_id", default="Xenium_human_Pancreas_FFPE")
    ap.add_argument("--ome", default="",
                    help="OME file OR the run directory")
    ap.add_argument("--zarr", default="")
    ap.add_argument("--out_root", default="/hpc/group/jilab/rz179/cellpt/nucleus_data")
    ap.add_argument("--channel_index", type=int, default=0, help="DAPI channel index")
    ap.add_argument("--out_size", type=int, default=224, help="Output canvas size")
    ap.add_argument("--target_mpp", type=float, default=0.50, help="Target microns per pixel for calibration")
    

    # Visual parameters
    ap.add_argument("--bg_type", default="median", choices=['median', 'black', 'noise'],
                    help="Background type for nucleus placement")
    ap.add_argument("--jitter", type=int, default=0, help="Random position jitter in pixels")
    
    # Processing parameters
    ap.add_argument("--pad_px", type=int, default=64, help="Padding around nucleus for window extraction")
    ap.add_argument("--tile", type=int, default=2048, help="Tile size for processing large images")
    ap.add_argument("--min_area", type=int, default=20, help="Minimum pixel area for initial filtering")
    ap.add_argument("--print_every", type=int, default=1000, help="Print progress every N nuclei")
    
    ap.add_argument("--celltype_csv", type=str, default="", help="Optional CSV with columns cellid,celltype")
    
    args = ap.parse_args()

    print(f"Processing slide: {args.slide_id}")
    print(f"Target MPP: {args.target_mpp} μm/pixel")
    print(f"Output size: {args.out_size}×{args.out_size} pixels")
    print(f"Background: {args.bg_type}")
    print(f"Jitter: ±{args.jitter} pixels")
    print()

    kept = process_nucleus_shapes_mpp(
        slide_id=args.slide_id,
        ome_or_dir=args.ome,
        zarr_path=args.zarr,
        out_root=args.out_root,
        channel_index=args.channel_index,
        out_size=args.out_size,
        target_mpp=args.target_mpp,
        bg_type=args.bg_type,
        jitter=args.jitter,
        pad_px=args.pad_px,
        tile=args.tile,
        min_area=args.min_area,
        print_every=args.print_every,
        celltype_csv=args.celltype_csv
    )
    
    print(f"Nucleus shape extraction complete!")
    print(f"Successfully extracted {kept} nucleus shapes with biological scale preservation")
    print(f"Output directory: {args.out_root}")

if __name__ == "__main__":
    main()