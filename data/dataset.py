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
    def __init__(self, csv_path, class_to_idx: Dict[str,int], mean, std, size, use_img_uniform=False, augment=False):
        df = pd.read_csv(csv_path)
        self.df = df.copy()

        # choose column
        if use_img_uniform:
            if "img_path.1" in df.columns:
                print(f"Using 'img_path.1' column from {csv_path} for images.")
                col = "img_path.1"
            else:
                raise RuntimeError(f"Column 'img_path.1' not found in {csv_path}. ")
        else:   
            col = "img_path"
        #col = "img_path.1" if use_img_uniform and "img_path.1" in df.columns else "img_path"
        if col not in df.columns:
            raise RuntimeError(f"Column '{col}' not found in {csv_path}")

        # keep only rows with existing files and allowed labels
        self.df = self.df[self.df[col].map(lambda p: Path(str(p)).is_file())]
        if "skipped_reason" in self.df.columns:
            self.df = self.df[self.df["skipped_reason"].fillna("") == ""]
        if "cell_type" not in self.df.columns:
            raise RuntimeError("CSV missing 'cell_type' column")

        allowed = set(class_to_idx.keys())
        self.df = self.df[self.df["cell_type"].isin(allowed)].reset_index(drop=True)

        self.path_col = col
        self.class_to_idx = class_to_idx

        # transforms
        self.size = size  
        mean = IMAGENET_DEFAULT_MEAN if mean is None else mean
        std  = IMAGENET_DEFAULT_STD  if std  is None else std
        
        erase_value = tuple((-(np.array(mean)/np.array(std))).tolist())
        
        base = [
            transforms.Resize((self.size, self.size), interpolation=IM.BICUBIC, antialias=True),
            transforms.ToTensor(),  # HWC [0..255] -> CHW [0..1]
            transforms.Lambda(lambda x: x.expand(3, *x.shape[1:]) if x.shape[0] == 1 else x),
            transforms.Normalize(mean, std),
        ]
        if augment:
            pre_aug = [
                #RandomCenterZoom(0.98, 1.00),

                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),

                # small rotation; fill black to match background
                transforms.RandomApply([
                    transforms.RandomRotation(8, interpolation=IM.BILINEAR, fill=0)
                ], p=0.5),

                # tiny translation/scale to simulate slight placement/size variation
                transforms.RandomApply([
                    transforms.RandomAffine(
                        degrees=0, translate=(0.05, 0.05), shear=0,
                        interpolation=IM.BILINEAR, fill=0
                    )
                ], p=0.5),

                transforms.ColorJitter(brightness=0.2, contrast=0.2),

                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
                ], p=0.2),
            ]
            # value=0 fills with the per-channel MEAN in normalized space (x=0),
            self.tx = transforms.Compose(
                pre_aug + base + [
                    transforms.RandomErasing(
                        p=0.25, scale=(0.02, 0.06), ratio=(0.3, 3.3), value=erase_value, inplace=True
                    )
                ]
            )
        else:
            self.tx = transforms.Compose(base)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row[self.path_col]).convert("L")  # grayscale
        x = self.tx(img)
        y = self.class_to_idx[row.cell_type]
        return x, y






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
