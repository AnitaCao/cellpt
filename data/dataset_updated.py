# data/dataset.py
from torch.utils.data import Dataset
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms import InterpolationMode as IM
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np
import random


class NucleusCSV(Dataset):
    """
    Two modes:
      - input_mode="raw": single grayscale image expanded to 3 channels and normalized by ImageNet stats
      - input_mode="rt" : 2 channels [raw, target_mask], where target_mask is binary in {0,1}
                          raw uses ImageNet gray stats, mask uses mean=0 std=1

    When return_meta=True, returns (x, y, meta) where meta includes:
      - "coarse_idx": int index of coarse label
      - "slide_idx":  int index of slide id
      - "slide":      slide string
    The dataset also provides:
      - coarse_to_idx: Dict[str,int]
      - fine_to_coarse_idx: List[int] aligned to class_to_idx order
    """
    def __init__(
        self,
        csv_path: str,
        class_to_idx: Dict[str, int],
        mean,
        std,
        size: int,
        label_col: str,
        use_img_uniform: bool = False,
        augment: bool = False,
        # two channel settings
        input_mode: str = "raw",  # "raw" or "rt"
        mask_target_col: str = "mask_target_img_path",
        mask_from_masked_images: bool = True,
        mask_threshold: int = 0,
        # metadata
        return_meta: bool = False,
        slide_col: str = "slide_id",
        coarse_col: str = "cell_type_coarse",
    ):
        super().__init__()
        self.input_mode = input_mode
        self.size = int(size)
        self.class_to_idx = class_to_idx
        self.label_col = label_col
        self.return_meta = bool(return_meta)
        self.slide_col = slide_col
        self.coarse_col = coarse_col
        self.mask_target_col = mask_target_col
        self.mask_from_masked_images = bool(mask_from_masked_images)
        self.mask_threshold = int(mask_threshold)

        df = pd.read_csv(csv_path)

        # choose path column
        if use_img_uniform and "img_path_uniform" in df.columns:
            path_col = "img_path_uniform"
        else:
            if "img_path" in df.columns:
                path_col = "img_path"
            elif "raw_img_path" in df.columns:
                path_col = "raw_img_path"
            else:
                raise RuntimeError("CSV missing image path column: need img_path, img_path_uniform, or raw_img_path")

        # basic filtering
        df = df[df[path_col].map(lambda p: Path(str(p)).is_file())]
        if "skipped_reason" in df.columns:
            df = df[df["skipped_reason"].fillna("") == ""]
        allowed = set(class_to_idx.keys())
        df = df[df[label_col].isin(allowed)]

        # in two channel mode ensure mask column exists and files exist
        if self.input_mode == "rt":
            if self.mask_target_col not in df.columns:
                raise RuntimeError(f"input_mode='rt' but column {self.mask_target_col} not found in CSV")
            df = df[df[self.mask_target_col].map(lambda p: Path(str(p)).is_file())]

        self.df = df.reset_index(drop=True)
        self.path_col = path_col

        # metadata maps
        if self.return_meta:
            if self.coarse_col not in self.df.columns or self.slide_col not in self.df.columns:
                raise RuntimeError(f"return_meta=True but {self.coarse_col} or {self.slide_col} not in CSV")
            self.coarse_to_idx = {c: i for i, c in enumerate(sorted(self.df[self.coarse_col].unique()))}
            self.slide_to_idx = {s: i for i, s in enumerate(sorted(self.df[self.slide_col].unique()))}
            # build fine to coarse index list aligned to class_to_idx order
            tmp = {row[self.label_col]: self.coarse_to_idx[row[self.coarse_col]] for _, row in self.df.iterrows()}
            self.fine_to_coarse_idx = [tmp[f] for f, _ in sorted(self.class_to_idx.items(), key=lambda kv: kv[1])]

        # normalization
        self.base_mean = IMAGENET_DEFAULT_MEAN if mean is None else mean
        self.base_std = IMAGENET_DEFAULT_STD if std is None else std

        # set up transforms
        if self.input_mode == "raw":
            self.tx = transforms.Compose([
                transforms.Resize((self.size, self.size), interpolation=IM.BICUBIC, antialias=True),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.expand(3, *x.shape[1:]) if x.shape[0] == 1 else x),
                transforms.Normalize(self.base_mean, self.base_std),
            ])
            self.aug_geom = None
            self.aug_color = None
            self.aug_blur = None
            if augment:
                self.tx = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomApply([transforms.RandomRotation(8, interpolation=IM.BILINEAR, fill=0)], p=0.5),
                    transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.05, 0.05),
                                                                    interpolation=IM.BILINEAR, fill=0)], p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
                    transforms.Resize((self.size, self.size), interpolation=IM.BICUBIC, antialias=True),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.expand(3, *x.shape[1:]) if x.shape[0] == 1 else x),
                    transforms.Normalize(self.base_mean, self.base_std),
                    transforms.RandomErasing(p=0.15, scale=(0.02, 0.06), ratio=(0.3, 3.3), value=0.0, inplace=True),
                ])
        else:
            # paired aug for raw and mask
            self.aug_geom = bool(augment)
            self.aug_color = transforms.ColorJitter(brightness=0.2, contrast=0.2) if augment else None
            self.aug_blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)) if augment else None
            # per channel mean and std for [raw, mask]
            m_raw = float(self.base_mean[0])
            s_raw = float(self.base_std[0])
            self.mean2 = [m_raw, 0.0]
            self.std2 = [s_raw, 1.0]

    def __len__(self):
        return len(self.df)

    def _paired_geom(self, raw: Image.Image, mt: Image.Image):
        # flips
        if random.random() < 0.5:
            raw = TF.hflip(raw)
            mt = TF.hflip(mt)
        if random.random() < 0.5:
            raw = TF.vflip(raw)
            mt = TF.vflip(mt)
        # rotation
        ang = random.uniform(-8, 8)
        raw = TF.rotate(raw, ang, interpolation=IM.BILINEAR, fill=0)
        mt = TF.rotate(mt, ang, interpolation=IM.NEAREST, fill=0)
        # small translation
        max_dx = 0.05 * raw.size[0]
        max_dy = 0.05 * raw.size[1]
        tx = int(random.uniform(-max_dx, max_dx))
        ty = int(random.uniform(-max_dy, max_dy))
        raw = TF.affine(raw, angle=0, translate=[tx, ty], scale=1.0, shear=[0, 0], interpolation=IM.BILINEAR, fill=0)
        mt = TF.affine(mt, angle=0, translate=[tx, ty], scale=1.0, shear=[0, 0], interpolation=IM.NEAREST, fill=0)
        return raw, mt

    def __getitem__(self, i):
        row = self.df.iloc[i]
        y = self.class_to_idx[row[self.label_col]]

        if self.input_mode == "raw":
            img = Image.open(row[self.path_col]).convert("L")
            x = self.tx(img)
            if not self.return_meta:
                return x, y
            meta = {
                "coarse_idx": self.coarse_to_idx[row[self.coarse_col]] if self.return_meta else -1,
                "slide_idx": self.slide_to_idx[row[self.slide_col]] if self.return_meta else -1,
                "slide": str(row[self.slide_col]) if self.return_meta else "",
            }
            return x, y, meta

        # two channel path
        raw = Image.open(row[self.path_col]).convert("L")
        mt = Image.open(row[self.mask_target_col]).convert("L")

        # derive binary mask if needed
        if self.mask_from_masked_images:
            arr = np.array(mt, dtype=np.uint8)
            bin_arr = (arr > self.mask_threshold).astype(np.uint8) * 255
            mt = Image.fromarray(bin_arr, mode="L")

        # paired geometry
        if self.aug_geom:
            raw, mt = self._paired_geom(raw, mt)

        # resize
        raw = TF.resize(raw, (self.size, self.size), interpolation=IM.BICUBIC)
        mt = TF.resize(mt, (self.size, self.size), interpolation=IM.NEAREST)

        # appearance on raw only
        if self.aug_color is not None:
            raw = self.aug_color(raw)
        if self.aug_blur is not None and random.random() < 0.2:
            raw = self.aug_blur(raw)

        # to tensors
        r = TF.to_tensor(raw)                 # [1,H,W], 0..1
        m = (TF.to_tensor(mt) > 0.5).float()  # [1,H,W], {0,1}

        x = torch.cat([r, m], dim=0)          # [2,H,W]
        x = TF.normalize(x, mean=self.mean2, std=self.std2)

        if not self.return_meta:
            return x, y
        meta = {
            "coarse_idx": self.coarse_to_idx[row[self.coarse_col]],
            "slide_idx": self.slide_to_idx[row[self.slide_col]],
            "slide": str(row[self.slide_col]),
        }
        return x, y, meta
