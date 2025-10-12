from datasets import load_from_disk
from torch.utils.data import Dataset
from PIL import Image
import torch
from typing import Dict, Optional
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import pandas as pd
import numpy as np
from torchvision import transforms
from pathlib import Path
from torchvision.transforms import InterpolationMode as IM
import torchvision.transforms.functional as TF
import random
import torch.nn as nn


class RandomCenterZoom:
    """Center-crop to a random fraction [min_zoom, max_zoom] of the image,
    then (later) you can Resize back to target size. Example: 0.90–1.00."""
    def __init__(self, min_zoom=0.90, max_zoom=1.00):
        assert 0 < min_zoom <= max_zoom <= 1.0
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom

    def __call__(self, img):
        w, h = img.size
        s = random.uniform(self.min_zoom, self.max_zoom)
        cw, ch = int(w * s), int(h * s)
        # center crop to (ch, cw)
        return TF.center_crop(img, (ch, cw))
    

class AddGaussianNoise(nn.Module):
    def __init__(self, sigma_min=0.01, sigma_max=0.03, p=0.3):
        super().__init__()
        self.sigma_min, self.sigma_max, self.p = sigma_min, sigma_max, p
    def forward(self, x):
        if random.random() < self.p:
            sigma = random.uniform(self.sigma_min, self.sigma_max)
            x = x + torch.randn_like(x) * sigma
            x = x.clamp(0, 1)
        return x


# -----------------------------
# Cell Core Dataset
# -----------------------------


class NucleusCSV(Dataset):
    def __init__(self, csv_path, class_to_idx: dict, mean, std, size,
                 label_col, augment=False,
                 return_slide: bool = False):
        df = pd.read_csv(csv_path) 
        path_col = "img_path"
        if path_col not in df.columns:
            path_col = "raw_img_path"
            if path_col not in df.columns:
                raise RuntimeError(f"CSV missing path column 'img_path' or 'raw_img_path'")

        print(f"Using path column '{path_col}' from {csv_path} for images.")
        
        if label_col not in df.columns:
            raise RuntimeError(f"CSV missing label column '{label_col}'")

        # keep rows with existing files and allowed labels
        df = df[df[path_col].map(lambda p: Path(str(p)).is_file())]
        if "skipped_reason" in df.columns:
            df = df[df["skipped_reason"].fillna("") == ""]

        allowed = set(class_to_idx.keys())
        df = df[df[label_col].isin(allowed)].reset_index(drop=True)

        # --- new: slide id return guard
        self.return_slide = bool(return_slide)
        if self.return_slide:
            if "slide_id" not in df.columns:
                raise RuntimeError(f"return_slide=True but column 'slide_id' not found in {csv_path}")

        # save fields
        self.df = df.copy()
        self.path_col = path_col
        self.label_col = label_col
        self.class_to_idx = class_to_idx

        # transforms (unchanged)
        self.size = size
        mean = IMAGENET_DEFAULT_MEAN if mean is None else mean
        std  = IMAGENET_DEFAULT_STD  if std  is None else std
        erase_value = tuple((-(np.array(mean)/np.array(std))).tolist())

        base = [
            transforms.Resize((self.size, self.size), interpolation=IM.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.expand(3, *x.shape[1:]) if x.shape[0] == 1 else x),
            transforms.Normalize(mean, std),
        ]

        if augment:
            pre_aug = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply([transforms.RandomRotation(8, interpolation=IM.BILINEAR, fill=0)], p=0.5),
                transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.05, 0.05),
                                                                interpolation=IM.BILINEAR, fill=0)], p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
            ]
            self.tx = transforms.Compose(
                pre_aug + base + [
                    transforms.RandomErasing(p=0.15, scale=(0.02, 0.06), ratio=(0.3, 3.3),
                                             value=0.0, inplace=True)
                ]
            )
        else:
            self.tx = transforms.Compose(base)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row[self.path_col]).convert("L")
        x = self.tx(img)
        y = self.class_to_idx[row[self.label_col]]
        if self.return_slide:
            sid = row["slide_id"]
            return x, y, sid
        return x, y


import numpy as np

class PairedGeomAug:
    """Sample one set of geometric params and apply to both raw and mask."""
    def __init__(self, rot=8, translate=0.05):
        self.rot = rot
        self.translate = translate
    def __call__(self, img, mask):
        # flips
        if random.random() < 0.5:
            img = TF.hflip(img); mask = TF.hflip(mask)
        if random.random() < 0.5:
            img = TF.vflip(img); mask = TF.vflip(mask)
        # rotation
        if self.rot > 0:
            ang = random.uniform(-self.rot, self.rot)
            img = TF.rotate(img, ang, interpolation=IM.BILINEAR, fill=0)
            mask = TF.rotate(mask, ang, interpolation=IM.NEAREST,  fill=0)
        # small translation
        if self.translate > 0:
            max_dx = self.translate * img.size[0]
            max_dy = self.translate * img.size[1]
            tx = int(random.uniform(-max_dx, max_dx))
            ty = int(random.uniform(-max_dy, max_dy))
            img = TF.affine(img, angle=0, translate=[tx, ty], scale=1.0, shear=[0,0],
                            interpolation=IM.BILINEAR, fill=0)
            mask = TF.affine(mask, angle=0, translate=[tx, ty], scale=1.0, shear=[0,0],
                             interpolation=IM.NEAREST, fill=0)
        return img, mask



class PairedMultiFOV(Dataset):
    """
    Returns:
        x: [2, C, H, W]  (view 0 = 2.5x, view 1 = 5x)
        y: int
        sid: slide_id (if present in left CSV; otherwise empty string)
    Pairing logic:
        - If key_col exists in both CSVs, join on it.
        - Else join by filename stem from path column(s).
    """
    def __init__(
        self,
        csv_a,                 # CSV for 2.5x
        csv_b,                 # CSV for 5x
        class_to_idx: dict,
        size: int = 224,
        mean=None, std=None,
        label_col: str = "cell_type",
        path_cols=("img_path", "raw_img_path", "img_path_uniform"),
        key_col: str = "cell_id",          # if not in both, will fall back to stem
        slide_col: str = "slide_id",
        augment: bool = False,
    ):
        self.size = int(size)
        self.label_col = label_col
        self.slide_col = slide_col
        self.class_to_idx = class_to_idx

        # Load CSVs
        A = pd.read_csv(csv_a).copy()
        B = pd.read_csv(csv_b).copy()

        # Decide path column per CSV (mirrors your NucleusCSV logic)
        def choose_path_col(df):
            for c in path_cols:
                if c in df.columns:
                    return c
            raise RuntimeError("No path column found among: " + ", ".join(path_cols))

        pa = choose_path_col(A)
        pb = choose_path_col(B)

        # Filter to existing files and allowed labels
        allowed = set(class_to_idx.keys())
        A = A[A[pa].map(lambda p: Path(str(p)).is_file())]
        B = B[B[pb].map(lambda p: Path(str(p)).is_file())]
        if "skipped_reason" in A.columns:
            A = A[A["skipped_reason"].fillna("") == ""]
        if "skipped_reason" in B.columns:
            B = B[B["skipped_reason"].fillna("") == ""]
        A = A[A[label_col].isin(allowed)]
        B = B[B[label_col].isin(allowed)]

        # Join by key_col if present in BOTH, else by filename stem
        join_on_key = key_col in A.columns and key_col in B.columns
        if join_on_key:
            merged = A[[key_col, pa, label_col, slide_col]].merge(
                B[[key_col, pb, label_col, slide_col]],
                on=key_col, suffixes=("_a", "_b")
            )
            keep = merged[f"{label_col}_a"] == merged[f"{label_col}_b"]
            self.join = merged.loc[keep].reset_index(drop=True)
            self.key_mode = "key"
            self.key_col = key_col
        else:
            # by stem
            A["__stem__"] = A[pa].map(lambda p: Path(str(p)).stem)
            B["__stem__"] = B[pb].map(lambda p: Path(str(p)).stem)
            merged = A[["__stem__", pa, label_col, slide_col]].merge(
                B[["__stem__", pb, label_col, slide_col]],
                on="__stem__", suffixes=("_a", "_b")
            )
            keep = merged[f"{label_col}_a"] == merged[f"{label_col}_b"]
            self.join = merged.loc[keep].reset_index(drop=True)
            self.key_mode = "stem"
            self.key_col = "__stem__"

        # Keep handy column names
        self.pa, self.pb = pa, pb
        
        nA, nB, nP = len(A), len(B), len(self.join)
        print(f"[PairedMultiFOV] paired={nP}  from A={nA} B={nB}")
        if nP == 0:
            raise RuntimeError("[PairedMultiFOV] No pairs found. Check key_col/stems, paths, or labels.")
        
        # --- expose a canonical df for trainers expecting NucleusCSV-like API
        label_a = f"{label_col}_a"
        slide_a = f"{slide_col}_a"
        cols = {}
        cols[label_col] = self.join[label_a].reset_index(drop=True)
        if slide_a in self.join.columns:
            cols["slide_id"] = self.join[slide_a].reset_index(drop=True)
        else:
            # fallback: empty strings if slide ids missing
            cols["slide_id"] = pd.Series([""] * len(self.join))
        self.df = pd.DataFrame(cols)

        # Transforms: same as your NucleusCSV, but we apply **paired** geom augs
        mean = IMAGENET_DEFAULT_MEAN if mean is None else mean
        std  = IMAGENET_DEFAULT_STD  if std  is None else std

        # geometric augments (paired)
        self.augment = augment
        self.geo = PairedGeomAug(rot=8, translate=0.05) if augment else None

        # color + resize + to tensor + norm (apply per-view)
        self.post = transforms.Compose([
            transforms.Resize((self.size, self.size), interpolation=IM.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.expand(3, *x.shape[1:]) if x.shape[0] == 1 else x),
            transforms.Normalize(mean, std),
        ])

        # Optional: light color jitter identically to both views (keeps alignment)
        self.color = transforms.ColorJitter(brightness=0.2, contrast=0.2) if augment else None

    def __len__(self):
        return len(self.join)

    def __getitem__(self, i):
        row = self.join.iloc[i]
        pa_col = f"{self.pa}_a"
        pb_col = f"{self.pb}_b"
        pa = row[pa_col]
        pb = row[pb_col]

        slide_a = f"{self.slide_col}_a"
        # load
        ia = Image.open(pa).convert("L")
        ib = Image.open(pb).convert("L")

        # paired geometric augments (keep the two views aligned)
        if self.augment and self.geo is not None:
            ia, ib = self.geo(ia, ib)

        # optional identical color jitter
        if self.augment and self.color is not None:
            # sample once, apply to both by using functional form
            # ColorJitter doesn't expose sampled params, so we apply the module twice with same seed
            s = torch.seed()
            torch.manual_seed(s); ia = self.color(ia)
            torch.manual_seed(s); ib = self.color(ib)

        xa = self.post(ia)  # [C,H,W]
        xb = self.post(ib)  # [C,H,W]
        x = torch.stack([xa, xb], dim=0)  # [2,C,H,W]

        y = self.class_to_idx[row[f"{self.label_col}_a"]]
        
        sid = row[slide_a] if slide_a in self.join.columns else ""

        return x, y, sid


























"""
# -----------------------------
class ArrowImageDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]
        label = example["label"]

        # If the image is stored as a PIL image in Hugging Face format, convert properly
        if isinstance(image, dict) and "bytes" in image:
            image = Image.open(image["path"])  # fallback
        elif hasattr(image, "convert"):  # Already a PIL image
            image = image.convert("RGB")
        else:
            raise ValueError("Unsupported image format")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)

def load_arrow_datasets(data_dir, transform_train=None, transform_val=None):
    from datasets import DatasetDict
    dataset = load_from_disk(data_dir)

    if isinstance(dataset, DatasetDict):
        train_ds = ArrowImageDataset(dataset["train"], transform=transform_train)
        val_ds   = ArrowImageDataset(dataset["validation"], transform=transform_val)
        test_ds  = ArrowImageDataset(dataset.get("test", dataset["validation"]), transform=transform_val)
    else:
        raise ValueError("Expected a DatasetDict with train/val/test splits.")

    return train_ds, val_ds, test_ds
"""