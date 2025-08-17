#!/usr/bin/env python3
import os, re, csv, argparse
from pathlib import Path
import numpy as np
import tifffile, zarr, cv2
from PIL import Image

def open_ome_as_zarr(ome_path, series=0, level=0):
    tf = tifffile.TiffFile(str(ome_path))
    store = tf.aszarr(series=series, level=level)
    z = zarr.open(store, mode="r")
    return tf, z

def get_pixel_size_um(ome_path):
    with tifffile.TiffFile(str(ome_path)) as tf:
        meta = tf.ome_metadata or ""
    rx = re.search(r'PhysicalSizeX="([\d\.eE+-]+)".*?PhysicalSizeY="([\d\.eE+-]+)"', meta)
    if not rx:
        return None, None
    return float(rx.group(1)), float(rx.group(2))

def open_nucseg_zarr(zarr_path, channel=0):
    g = zarr.open(str(zarr_path), mode="r")
    arr = g["masks"][channel]  # do not np.asarray
    return arr  # zarr.Array, shape (H, W), dtype int32/64

def percentile_norm(img, p1=1.0, p99=99.0):
    lo, hi = np.percentile(img, p1), np.percentile(img, p99)
    hi = max(hi, lo + 1e-6)
    img = np.clip(img, lo, hi)
    return ((img - lo) / (hi - lo) * 255.0).astype(np.uint8)

def ensure_csv(path, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    f = open(path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=header)
    if new:
        w.writeheader()
    return f, w

def tile_ranges(H, W, tile):
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            y1 = min(H, y0 + tile)
            x1 = min(W, x0 + tile)
            yield y0, y1, x0, x1

def build_index_streaming(lbl_z, tile=2048, min_area=20, max_id_hint=None):
    H, W = lbl_z.shape
    # first pass: find max id if not provided
    max_id = 0
    if max_id_hint is None:
        for y0, y1, x0, x1 in tile_ranges(H, W, tile):
            chunk = np.asarray(lbl_z[y0:y1, x0:x1])
            if chunk.size == 0:
                continue
            m = int(chunk.max())
            if m > max_id:
                max_id = m
    else:
        max_id = int(max_id_hint)

    # per id accumulators
    count = np.zeros(max_id + 1, dtype=np.int64)
    sum_x = np.zeros(max_id + 1, dtype=np.float64)
    sum_y = np.zeros(max_id + 1, dtype=np.float64)
    min_x = np.full(max_id + 1, np.inf, dtype=np.float64)
    min_y = np.full(max_id + 1, np.inf, dtype=np.float64)
    max_x = np.full(max_id + 1, -np.inf, dtype=np.float64)
    max_y = np.full(max_id + 1, -np.inf, dtype=np.float64)

    # second pass: accumulate stats
    for y0, y1, x0, x1 in tile_ranges(H, W, tile):
        chunk = np.asarray(lbl_z[y0:y1, x0:x1])
        if chunk.size == 0:
            continue
        ids = np.unique(chunk)
        ids = ids[ids != 0]
        if ids.size == 0:
            continue

        for cid in ids:
            m = (chunk == cid)
            n = int(m.sum())
            if n == 0:
                continue
            ys, xs = np.nonzero(m)
            # global coords
            xs_g = xs + x0
            ys_g = ys + y0
            count[cid] += n
            sum_x[cid] += xs_g.sum()
            sum_y[cid] += ys_g.sum()
            # bbox updates
            cx_min = xs_g.min(); cx_max = xs_g.max()
            cy_min = ys_g.min(); cy_max = ys_g.max()
            if cx_min < min_x[cid]: min_x[cid] = cx_min
            if cy_min < min_y[cid]: min_y[cid] = cy_min
            if cx_max > max_x[cid]: max_x[cid] = cx_max
            if cy_max > max_y[cid]: max_y[cid] = cy_max

    # build list of nuclei
    nuclei = []
    for cid in range(1, max_id + 1):
        n = int(count[cid])
        if n < min_area:
            continue
        cx = int(round(sum_x[cid] / n))
        cy = int(round(sum_y[cid] / n))
        if not np.isfinite(min_x[cid]):
            continue
        x0 = int(min_x[cid]); y0 = int(min_y[cid])
        x1 = int(max_x[cid]); y1 = int(max_y[cid])
        w = x1 - x0 + 1; h = y1 - y0 + 1
        nuclei.append({
            "cid": cid, "area_px": n, "centroid": (cx, cy), "bbox": (x0, y0, w, h)
        })
    return nuclei, (H, W)

def clamp_center_box(cx, cy, side, H, W):
    need_pad = side > W or side > H
    x0 = max(0, min(cx - side // 2, max(0, W - side)))
    y0 = max(0, min(cy - side // 2, max(0, H - side)))
    return int(x0), int(y0), int(side), bool(need_pad)

def read_window_with_padding(zarr_img, channel_index, x0, y0, side):
    if len(zarr_img.shape) == 3:
        _, H, W = zarr_img.shape
        patch = np.asarray(zarr_img[channel_index, max(0,y0):min(H,y0+side), max(0,x0):min(W,x0+side)])
    else:
        H, W = zarr_img.shape
        patch = np.asarray(zarr_img[max(0,y0):min(H,y0+side), max(0,x0):min(W,x0+side)])
    # pad if needed
    pad_top = max(0, -y0)
    pad_left = max(0, -x0)
    pad_bottom = max(0, (y0+side) - H)
    pad_right = max(0, (x0+side) - W)
    if pad_top or pad_left or pad_bottom or pad_right:
        patch = cv2.copyMakeBorder(patch, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    return patch

def slice_label_with_padding(lbl_z, x0, y0, side):
    H, W = lbl_z.shape
    crop = np.asarray(lbl_z[max(0,y0):min(H,y0+side), max(0,x0):min(W,x0+side)])
    pad_top = max(0, -y0)
    pad_left = max(0, -x0)
    pad_bottom = max(0, (y0+side) - H)
    pad_right = max(0, (x0+side) - W)
    if pad_top or pad_left or pad_bottom or pad_right:
        crop = cv2.copyMakeBorder(crop.astype(np.int32), pad_top, pad_bottom, pad_left, pad_right,
                                  borderType=cv2.BORDER_CONSTANT, value=0)
    return crop

def process_slide(slide_id, ome_path, zarr_path, out_root, split, fov_um,
                  pixel_size_um=None, channel_index=0, min_area=20, out_size=224,
                  tile=2048, print_every=2000):
    out_root = Path(out_root)
    img_dir = out_root / split / slide_id / "img"
    msk_dir = out_root / split / slide_id / "mask"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)
    meta_csv = out_root / "meta" / "samples.csv"
    header = [
        "slide_id","split","nucleus_id","patch_path","mask_path",
        "crop_x","crop_y","crop_side_px","pixel_size_um","fov_um",
        "area_px","area_um2","coverage","neighbor_frac","centroid_x","centroid_y"
    ]
    mf, writer = ensure_csv(meta_csv, header)

    if pixel_size_um is None:
        px_x, px_y = get_pixel_size_um(ome_path)
        if px_x is None:
            raise RuntimeError(f"No pixel size in OME: {ome_path}")
        if abs(px_x - px_y) > 1e-6:
            print(f"Warning: non-square pixels X {px_x} Y {px_y}. Using X.")
        pixel_size_um = float(px_x)
    side_px = int(round(fov_um / pixel_size_um))

    # open image and label lazily
    tf, zimg = open_ome_as_zarr(ome_path)
    lbl_z = open_nucseg_zarr(zarr_path, channel=0)
    H, W = lbl_z.shape
    # check shapes with image zarr
    zshape = zimg.shape
    Hy, Wx = (zshape[1], zshape[2]) if len(zshape) == 3 else (zshape[0], zshape[1])
    if (Hy, Wx) != (H, W):
        tf.close()
        raise RuntimeError(f"Shape mismatch. OME {Hy, Wx}, Label {H, W}")

    # pass 1: build nucleus index without loading full label
    nuclei, _ = build_index_streaming(lbl_z, tile=tile, min_area=min_area)
    n = len(nuclei)
    kept = 0

    # pass 2: write patches
    for k, info in enumerate(nuclei, 1):
        cid = info["cid"]
        cx, cy = info["centroid"]
        area_px = info["area_px"]

        x0, y0, side, _ = clamp_center_box(cx, cy, side_px, H, W)
        win = read_window_with_padding(zimg, channel_index, x0, y0, side)
        win_u8 = percentile_norm(win)

        crop_ids = slice_label_with_padding(lbl_z, x0, y0, side)
        mask_this = (crop_ids == cid).astype(np.uint8) * 255
        mask_other = ((crop_ids != 0) & (crop_ids != cid)).astype(np.uint8)
        neighbor_frac = float(mask_other.sum()) / float(side * side)
        coverage = float(area_px) / float(side * side)

        img224 = cv2.resize(win_u8, (out_size, out_size), interpolation=cv2.INTER_AREA)
        msk224 = cv2.resize(mask_this, (out_size, out_size), interpolation=cv2.INTER_NEAREST)

        img_name = f"{slide_id}_n{int(cid)}.png"
        msk_name = f"{slide_id}_n{int(cid)}_mask.png"
        cv2.imwrite(str(img_dir / img_name), img224)
        Image.fromarray(msk224).save(msk_dir / msk_name)

        area_um2 = float(area_px) * (pixel_size_um ** 2)
        writer.writerow({
            "slide_id": slide_id,
            "split": split,
            "nucleus_id": int(cid),
            "patch_path": str((img_dir / img_name).resolve()),
            "mask_path": str((msk_dir / msk_name).resolve()),
            "crop_x": int(x0),
            "crop_y": int(y0),
            "crop_side_px": int(side),
            "pixel_size_um": float(pixel_size_um),
            "fov_um": float(fov_um),
            "area_px": int(area_px),
            "area_um2": float(area_um2),
            "coverage": float(coverage),
            "neighbor_frac": float(neighbor_frac),
            "centroid_x": int(cx),
            "centroid_y": int(cy),
        })
        kept += 1
        if k % print_every == 0:
            print(f"{slide_id}: {k}/{n} processed, {kept} kept")

    mf.close()
    tf.close()
    print(f"{slide_id}: done. kept {kept} of {n} nuclei")
    return kept

def run_single(args):
    process_slide(
        slide_id=args.slide_id,
        ome_path=args.ome,
        zarr_path=args.zarr,
        out_root=args.out_root,
        split=args.split,
        fov_um=args.fov_um,
        pixel_size_um=args.pixel_size_um,
        channel_index=args.channel_index,
        min_area=args.min_area,
        out_size=args.out_size,
        tile=args.tile,
        print_every=args.print_every
    )

def run_manifest(args):
    rows = list(csv.DictReader(open(args.manifest)))
    total = 0
    for r in rows:
        slide_id = r["slide_id"]
        split = r.get("split") or args.default_split
        kept = process_slide(
            slide_id=slide_id,
            ome_path=r["ome_path"],
            zarr_path=r["zarr_path"],
            out_root=args.out_root,
            split=split,
            fov_um=args.fov_um,
            pixel_size_um=float(r["pixel_size_x_um"]) if r.get("pixel_size_x_um") else None,
            channel_index=args.channel_index,
            min_area=args.min_area,
            out_size=args.out_size,
            tile=args.tile,
            print_every=args.print_every
        )
        total += kept
    print(f"All slides done. total kept {total}")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    s1 = sub.add_parser("single")
    s1.add_argument("--slide_id", required=True)
    s1.add_argument("--ome", required=True)
    s1.add_argument("--zarr", required=True)
    s1.add_argument("--out_root", required=True)
    s1.add_argument("--split", default="train")
    s1.add_argument("--fov_um", type=float, default=48.0)
    s1.add_argument("--pixel_size_um", type=float, default=None)
    s1.add_argument("--channel_index", type=int, default=0)
    s1.add_argument("--min_area", type=int, default=20)
    s1.add_argument("--out_size", type=int, default=224)
    s1.add_argument("--tile", type=int, default=2048)
    s1.add_argument("--print_every", type=int, default=2000)

    s2 = sub.add_parser("manifest")
    s2.add_argument("--manifest", required=True)
    s2.add_argument("--out_root", required=True)
    s2.add_argument("--default_split", default="train")
    s2.add_argument("--fov_um", type=float, default=48.0)
    s2.add_argument("--channel_index", type=int, default=0)
    s2.add_argument("--min_area", type=int, default=20)
    s2.add_argument("--out_size", type=int, default=224)
    s2.add_argument("--tile", type=int, default=2048)
    s2.add_argument("--print_every", type=int, default=2000)

    args = ap.parse_args()
    if args.mode == "single":
        run_single(args)
    else:
        run_manifest(args)

if __name__ == "__main__":
    main()
