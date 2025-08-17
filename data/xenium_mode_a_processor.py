#!/usr/bin/env python3
import os, re, csv, argparse, json
from pathlib import Path
import numpy as np
import tifffile, zarr, cv2
from PIL import Image
import sys

# ---------- helpers ----------
def open_ome_as_zarr(ome_path, series=0, level=0):
    tf = tifffile.TiffFile(str(ome_path))
    store = tf.aszarr(series=series, level=level)   # lazy
    z = zarr.open(store, mode="r")                  # (C,Y,X) or (Y,X)
    return tf, z

def open_nucseg_zarr(zarr_path, channel=0):
    g = zarr.open(str(zarr_path), mode="r")
    
    # Direct access for Xenium data
    masks = g['masks']
    
    # Use channel 0 (nucleus) or 1 (cell) segmentation
    key = str(channel)  # Convert 0 -> "0", 1 -> "1"
    
    if key in masks:
        mask = masks[key]
        print(f"Using masks['{key}'] with shape {mask.shape} and dtype {mask.dtype}")
        return mask
    else:
        available = list(masks.keys())
        raise RuntimeError(f"Could not find masks['{key}'] in zarr file. Available: {available}")

def auto_pick_ome(ome_or_dir, zarr_path):
    p = Path(ome_or_dir)
    
    # If it's a file, use it directly without dimension checking
    if p.is_file():
        print(f"Using specified OME file: {p}")
        sys.stdout.flush()

        return str(p)
    
    # Only do auto-detection if a directory is provided
    lbl = open_nucseg_zarr(zarr_path, channel=0)
    H, W = lbl.shape
    print(f"Looking for OME file matching label dimensions: {H} x {W}")
    
    cands = sorted((p/"morphology_focus").glob("morphology_focus_*.ome.tif"))
    if not cands and (p/"morphology_focus.ome.tif").exists():
        cands = [p/"morphology_focus.ome.tif"]
    
    print(f"Found {len(cands)} candidate OME files")
    
    for cand in cands:
        print(f"Checking {cand}...")
        try:
            with tifffile.TiffFile(str(cand)) as tf:
                z = zarr.open(tf.aszarr(series=0, level=0), mode="r")
                Hy, Wx = (z.shape[1], z.shape[2]) if len(z.shape)==3 else (z.shape[0], z.shape[1])
                print(f"  OME dimensions: {Hy} x {Wx}")
                if (Hy, Wx) == (H, W):
                    print(f"  ✓ Match found!")
                    return str(cand)
                else:
                    print(f"  ✗ No match")
        except Exception as e:
            print(f"  Error reading {cand}: {e}")
    
    raise RuntimeError(f"No OME tile matches label shape {H}x{W} in {p}")

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
    """streaming centroid + bbox per id"""
    H, W = lbl_z.shape
    print(f"Image size: {H} x {W} pixels")
    sys.stdout.flush()
    max_id = 0
    for y0,y1,x0,x1 in tile_ranges(H,W,tile):
        c = np.asarray(lbl_z[y0:y1, x0:x1])
        if c.size: max_id = max(max_id, int(c.max()))
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
        if npx[cid] < min_area or not np.isfinite(min_x[cid]): continue
        cx = int(round(sum_x[cid]/npx[cid])); cy = int(round(sum_y[cid]/npx[cid]))
        x0, y0 = int(min_x[cid]), int(min_y[cid])
        x1, y1 = int(max_x[cid]), int(max_y[cid])
        nuclei.append({"cid": cid, "area_px": int(npx[cid]),
                       "centroid": (cx,cy), "bbox": (x0,y0, x1-x0+1, y1-y0+1)})
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

def synth_background(local_img, local_lbl, cid, out_size=224):
    bg_mask = (local_lbl != cid)
    vals = local_img[bg_mask] if bg_mask.any() else local_img.reshape(-1)
    base = np.percentile(vals, 10)
    noise = np.random.normal(0, 2, (out_size, out_size))
    return np.clip(base + noise, 0, 255).astype(np.uint8)

def paste_unscaled_nucleus(win_img_u8, win_lbl_ids, cid, target=224, jitter=3):
    m = (win_lbl_ids == cid)
    if not m.any(): return None
    ys, xs = np.nonzero(m)
    y0,y1 = ys.min(), ys.max()
    x0,x1 = xs.min(), xs.max()
    sub_img = win_img_u8[y0:y1+1, x0:x1+1]
    sub_msk = m[y0:y1+1, x0:x1+1].astype(np.uint8)*255
    H, W = sub_img.shape
    if H>target or W>target:
        return None
    bg = synth_background(win_img_u8, win_lbl_ids, cid, out_size=target)
    oy = target//2 - H//2 + np.random.randint(-jitter, jitter+1)
    ox = target//2 - W//2 + np.random.randint(-jitter, jitter+1)
    oy = np.clip(oy, 0, target-H); ox = np.clip(ox, 0, target-W)
    out  = bg.copy()
    outm = np.zeros((target,target), np.uint8)
    reg = out[oy:oy+H, ox:ox+W]
    out[oy:oy+H, ox:ox+W] = np.where(sub_msk.astype(bool), sub_img, reg)
    outm[oy:oy+H, ox:ox+W] = sub_msk
    cov = float((sub_msk>0).sum())/float(target*target)
    return out, outm, cov, (H,W)

# ---------- main Mode A ----------
def process_modeA(slide_id, ome_or_dir, zarr_path, out_root,
                  channel_index=0, out_size=224,
                  coverage_min=0.10, coverage_max=0.70,
                  pad_px=64, tile=2048, min_area=20, print_every=2000):
    print("Starting Mode A processing...")
    
    out_root = Path(out_root)
    img_dir = out_root / "all" / slide_id / "img"
    msk_dir = out_root / "all" / slide_id / "mask"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    meta = out_root / "meta" / "metadata_modeA.csv"
    meta.parent.mkdir(parents=True, exist_ok=True)
    new = not meta.exists()
    cols = ["slide_id","nucleus_id","img_path","mask_path",
            "orig_bbox_w_px","orig_bbox_h_px","orig_area_px",
            "coverage","patch_h_px","patch_w_px"]
    f = open(meta, "a", newline=""); wr = csv.DictWriter(f, fieldnames=cols)
    if new: wr.writeheader()

    ome_path = auto_pick_ome(ome_or_dir, zarr_path)
    print(f"Using OME file: {ome_path}")
    sys.stdout.flush()
    
    tf, zimg = open_ome_as_zarr(ome_path)
    lbl_z = open_nucseg_zarr(zarr_path, channel=0)

    H, W = lbl_z.shape
    zshape = zimg.shape
    Hy, Wx = (zshape[1], zshape[2]) if len(zshape)==3 else (zshape[0], zshape[1])
    if (Hy, Wx)!=(H,W):
        tf.close(); f.close()
        raise RuntimeError(f"Shape mismatch OME {Hy,Wx} vs Label {H,W}")

    print(f"Building nucleus index...")
    
    nuclei, _ = build_nucleus_index(lbl_z, tile=tile, min_area=min_area)
    n = len(nuclei); kept=0
    print(f"Found {n} nuclei to process")
    sys.stdout.flush()
    
    skip_big = skip_cov = 0
    for k, info in enumerate(nuclei, 1):
        cid = info["cid"]
        (cx,cy) = info["centroid"]
        (bx,by,bw,bh) = info["bbox"]
        area_px = info["area_px"]

        # local window around bbox + pad for bg sampling
        side = max(bw,bh) + 2*pad_px
        x0 = max(0, min(cx - side//2, W - side))
        y0 = max(0, min(cy - side//2, H - side))

        win = read_window_with_padding(zimg, channel_index, x0, y0, side)
        win_u8 = percentile_norm(win)
        crop_ids = slice_label_with_padding(lbl_z, x0, y0, side)

        pasted = paste_unscaled_nucleus(win_u8, crop_ids, cid, target=out_size, jitter=3)
        if pasted is None:
            skip_big += 1  # too big to paste unscaled (H or W > target) or no mask
            continue
        img224, msk224, cov, (ph,pw) = pasted
        if cov < coverage_min or cov > coverage_max:
            skip_cov += 1
            continue

        img_name = f"{slide_id}_n{cid}.png"
        msk_name = f"{slide_id}_n{cid}_mask.png"
        cv2.imwrite(str(img_dir / img_name), img224)
        Image.fromarray(msk224).save(msk_dir / msk_name)

        wr.writerow({
            "slide_id": slide_id,
            "nucleus_id": int(cid),
            "img_path": str((img_dir / img_name).resolve()),
            "mask_path": str((msk_dir / msk_name).resolve()),
            "orig_bbox_w_px": int(bw),
            "orig_bbox_h_px": int(bh),
            "orig_area_px": int(area_px),
            "coverage": float(cov),
            "patch_h_px": int(ph),
            "patch_w_px": int(pw)
        })
        kept+=1
        if k % print_every == 0:
            print(f"{slide_id}: {k}/{n} processed, {kept} kept")

    print(f"{slide_id}: kept {kept}/{n} | skipped_big={skip_big} | skipped_cov={skip_cov}")

    f.close(); tf.close()
    print(f"{slide_id}: Mode A done. kept {kept}/{n}")
    return kept

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide_id", default="Xenium_human_Pancreas_FFPE")
    ap.add_argument("--ome", default="/hpc/group/jilab/rz179/XeniumData/Xenium_human_Pancreas_FFPE", help="OME file OR the run directory")
    ap.add_argument("--zarr", default="/hpc/group/jilab/rz179/XeniumData/Xenium_human_Pancreas_FFPE/cells.zarr.zip")
    ap.add_argument("--out_root", default="/hpc/group/jilab/rz179/cellpt/data/Xenium_human_Pancreas_FFPE")
    ap.add_argument("--channel_index", type=int, default=0)  # DAPI
    ap.add_argument("--out_size", type=int, default=224)
    ap.add_argument("--coverage_min", type=float, default=0.01)
    ap.add_argument("--coverage_max", type=float, default=0.95)
    ap.add_argument("--pad_px", type=int, default=64)
    ap.add_argument("--tile", type=int, default=2048)
    ap.add_argument("--min_area", type=int, default=5)
    args = ap.parse_args()

    process_modeA(
        slide_id=args.slide_id,
        ome_or_dir=args.ome,
        zarr_path=args.zarr,
        out_root=args.out_root,
        channel_index=args.channel_index,
        out_size=args.out_size,
        coverage_min=args.coverage_min,
        coverage_max=args.coverage_max,
        pad_px=args.pad_px,
        tile=args.tile,
        min_area=args.min_area
    )

if __name__ == "__main__":
    main()