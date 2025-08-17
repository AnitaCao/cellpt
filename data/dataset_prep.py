import os
import argparse
import numpy as np
import tifffile, zarr, cv2, re


"""
# Function to extract pixel size from OME-TIFF metadata ---------------------------------------------
def get_pixel_size_um(ome_tiff_path):
    with tifffile.TiffFile(ome_tiff_path) as tf:
        axes = tf.series[0].axes
        shape = tf.series[0].shape
        meta = tf.ome_metadata or ""
    # Try to read PhysicalSizeX and PhysicalSizeY from the OME XML
    rx = re.search(r'PhysicalSizeX="([\d\.eE+-]+)"\s+PhysicalSizeXUnit="(\w+)"', meta)
    ry = re.search(r'PhysicalSizeY="([\d\.eE+-]+)"\s+PhysicalSizeYUnit="(\w+)"', meta)
    px_x = float(rx.group(1)) if rx else None
    unit_x = rx.group(2) if rx else None
    px_y = float(ry.group(1)) if ry else None
    unit_y = ry.group(2) if ry else None
    print("Axes:", axes, "Shape:", shape)
    print("Pixel size X:", px_x, unit_x, "Pixel size Y:", px_y, unit_y)
    return (px_x, px_y, unit_x, unit_y)


get_pixel_size_um("/hpc/group/jilab/rz179/XeniumData/Xenium_human_Brain_GBM_FFPE/morphology_focus/morphology_focus_0000.ome.tif")


#!/usr/bin/env python3
import os
import argparse
import numpy as np
import tifffile, zarr, cv2

def read_nucseg(zarr_path, used_lbl_channel=0):
    store = zarr.ZipStore(zarr_path, mode="r") if str(zarr_path).endswith(".zip") else zarr.DirectoryStore(zarr_path)
    g = zarr.group(store=store)
    lbl = np.asarray(g["masks"][used_lbl_channel]).squeeze().astype(np.int32)
    return lbl

def read_ome_channel(ome_path, channel_index=0):
    # We only need shape and pixel size from OME. Image pixel values are not used for stats.
    with tifffile.TiffFile(ome_path) as tf:
        arr = tf.asarray()
        axes = tf.series[0].axes  # e.g., CYX
        meta = tf.ome_metadata or ""
    if "C" in axes:
        c_axis = axes.index("C")
        # sanity-slice a channel to verify shapes match nucseg later if needed
        sl = [slice(None)] * arr.ndim
        sl[c_axis] = channel_index
        img = np.squeeze(arr[tuple(sl)])
    else:
        img = np.squeeze(arr)
    # parse pixel size in microns
    def parse_px(m):
        import re
        rx = re.search(r'PhysicalSizeX="([\d\.eE+-]+)".*?PhysicalSizeY="([\d\.eE+-]+)"', m)
        if not rx:
            return None, None
        return float(rx.group(1)), float(rx.group(2))
    px_um_x, px_um_y = parse_px(meta)
    return img, px_um_x, px_um_y

def nucleus_bbox(mask_bool):
    ys, xs = np.nonzero(mask_bool)
    if ys.size == 0:
        return None
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    return int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)

def nucleus_centroid(mask_bool):
    M = cv2.moments(mask_bool.astype(np.uint8))
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy

def square_crop_box(cx, cy, side, H, W):
    x0 = max(0, min(cx - side // 2, W - side))
    y0 = max(0, min(cy - side // 2, H - side))
    x1, y1 = x0 + side, y0 + side
    # compute padding needed
    pad_top = max(0, -y0)
    pad_left = max(0, -x0)
    pad_bottom = max(0, y1 - H)
    pad_right = max(0, x1 - W)
    # clamp box to image
    xi0, yi0 = max(0, y0), max(0, x0)
    xi1, yi1 = min(H, y1), min(W, x1)
    return x0, y0, side, (pad_top + pad_bottom + pad_left + pad_right) > 0

def stats_for_fov(lbl, fov_um, px_um, max_n=20000, seed=0):
  
    #lbl: HxW int32 instance mask
    #fov_um: float microns
    #px_um: float microns per pixel (assume square pixels)
   # returns dict of metrics

    H, W = lbl.shape
    ids = np.unique(lbl)
    ids = ids[ids != 0]
    rng = np.random.default_rng(seed)
    if len(ids) > max_n:
        ids = rng.choice(ids, size=max_n, replace=False)

    side_px = int(round(fov_um / px_um))
    crop_area = side_px * side_px

    coverages = []
    neighbor_fracs = []
    border_flags = []
    kept = 0

    for cid in ids:
        mask = (lbl == cid)
        cen = nucleus_centroid(mask)
        if cen is None:
            continue
        cx, cy = cen
        x0, y0, side, at_border = square_crop_box(cx, cy, side_px, H, W)
        # slice crop within bounds
        xi0, yi0 = max(0, x0), max(0, y0)
        xi1, yi1 = min(W, x0 + side), min(H, y0 + side)
        crop_ids = lbl[yi0:yi1, xi0:xi1]
        # count pixels of this nucleus and others within the crop
        this_px = np.count_nonzero(crop_ids == cid)
        other_px = np.count_nonzero((crop_ids != 0) & (crop_ids != cid))
        cov = this_px / float(crop_area)
        neigh = other_px / float(crop_area)
        coverages.append(cov)
        neighbor_fracs.append(neigh)
        border_flags.append(1 if at_border else 0)
        kept += 1

    if kept == 0:
        return {
            "fov_um": fov_um,
            "side_px": side_px,
            "n": 0
        }

    coverages = np.asarray(coverages)
    neighbor_fracs = np.asarray(neighbor_fracs)
    border_flags = np.asarray(border_flags)

    def pct(a, q): return float(np.percentile(a, q))
    return {
        "fov_um": fov_um,
        "side_px": side_px,
        "n": int(kept),
        "coverage_mean": float(coverages.mean()),
        "coverage_p10": pct(coverages, 10),
        "coverage_p50": pct(coverages, 50),
        "coverage_p90": pct(coverages, 90),
        "neighbor_mean": float(neighbor_fracs.mean()),
        "neighbor_p90": pct(neighbor_fracs, 90),
        "border_clip_rate": float(border_flags.mean())
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ome", default="/hpc/group/jilab/rz179/XeniumData/Xenium_human_Brain_GBM_FFPE/morphology_focus/morphology_focus_0000.ome.tif", help="path to morphology_focus_0000.ome.tif")
    ap.add_argument("--zarr", default="/hpc/group/jilab/rz179/XeniumData/Xenium_human_Brain_GBM_FFPE/cells.zarr.zip", help="path to cells.zarr.zip")
    ap.add_argument("--nuc_channel", type=int, default=0, help="0 for nucseg")
    ap.add_argument("--max_n", type=int, default=30000, help="max nuclei to sample for stats")
    ap.add_argument("--fovs", type=float, nargs="+", default=[40.0, 48.0, 64.0], help="micron FOVs to test")
    args = ap.parse_args()

    # load
    lbl = read_nucseg(args.zarr, used_lbl_channel=args.nuc_channel)
    img, px_um_x, px_um_y = read_ome_channel(args.ome, channel_index=0)
    if lbl.shape != img.shape:
        print(f"Warning: shape mismatch image {img.shape} vs label {lbl.shape}. Continuing with label shape only.")
    if px_um_x is None or px_um_y is None:
        raise RuntimeError("Pixel size missing in OME. Cannot compute fixed microns FOV.")
    if abs(px_um_x - px_um_y) > 1e-6:
        print(f"Warning: non-square pixels X {px_um_x} Y {px_um_y}. Using X.")

    px_um = float(px_um_x)
    print(f"Pixel size: {px_um:.6f} um per pixel")

    rows = []
    for fov in args.fovs:
        s = stats_for_fov(lbl, fov, px_um, max_n=args.max_n)
        rows.append(s)

    # pretty print
    hdr = ["fov_um","side_px","n","coverage_mean","coverage_p10","coverage_p50","coverage_p90","neighbor_mean","neighbor_p90","border_clip_rate"]
    print("\nResults")
    print("\t".join(hdr))
    for r in rows:
        print("\t".join(str(r.get(k, "")) for k in hdr))

if __name__ == "__main__":
    main()
    



#---------------Check classification columns in cells.csv.gz-------------------
import pandas as pd

# Load your cells.csv.gz
df = pd.read_csv("/hpc/group/jilab/rz179/XeniumData/Xenium_human_Brain_GBM_FFPE/cells.csv.gz")

# Check column names
print("Available columns:")
print(df.columns.tolist())

# Look for classification columns
class_cols = [col for col in df.columns if any(keyword in col.lower() 
             for keyword in ['cell_type', 'cluster', 'class', 'label', 'type'])]
print(f"\nClassification columns found: {class_cols}")

# If found, show sample values
if class_cols:
    print(f"\nSample values from {class_cols[0]}:")
    print(df[class_cols[0]].value_counts().head(10))
    

#---------------------------------------------------------------------------------

#---------------Check cells.zarr.zip metadata-------------------------------------

    
    
import zarr
import numpy as np

# Load the cell_summary data
store = zarr.ZipStore('/hpc/group/jilab/rz179/XeniumData/Xenium_human_Brain_GBM_FFPE/cells.zarr.zip', mode='r')
z = zarr.group(store=store)

# Check cell_summary contents
cell_summary = z['cell_summary']
print(f"Cell summary shape: {cell_summary.shape}")
print(f"Cell summary dtype: {cell_summary.dtype}")

# Look at first few rows to understand the data
sample_data = np.array(cell_summary[:10])
print(f"Sample data (first 10 cells):")
print(sample_data)

# Check data ranges for each column
print(f"\nData ranges for each column:")
for i in range(cell_summary.shape[1]):
    col_data = np.array(cell_summary[:1000, i])  # Sample 1000 cells
    print(f"Column {i}: min={col_data.min():.2f}, max={col_data.max():.2f}, mean={col_data.mean():.2f}")
    
 """   
#!/usr/bin/env python3
import zarr
import tifffile
import sys

print("Testing file access...")
sys.stdout.flush()

# Test zarr
try:
    z = zarr.open('/hpc/group/jilab/rz179/XeniumData/Xenium_human_Lymph_Node_FFPE/cells.zarr.zip')
    print(f"Zarr keys: {list(z.keys())}")
    if 'masks' in z:
        print(f"Masks keys: {list(z['masks'].keys())}")
    sys.stdout.flush()
except Exception as e:
    print(f"Zarr error: {e}")
    sys.stdout.flush()

# Test OME files
try:
    import os
    morph_dir = '/hpc/group/jilab/rz179/XeniumData/Xenium_human_Lymph_Node_FFPE/morphology_focus'
    files = os.listdir(morph_dir)
    print(f"Morphology files: {files}")
    sys.stdout.flush()
except Exception as e:
    print(f"OME directory error: {e}")
    sys.stdout.flush()

print("Test complete")