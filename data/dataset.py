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

# -----------------------------
# Cell Core Dataset
# -----------------------------
class NucleusCSV(Dataset):
    def __init__(self, csv_path, class_to_idx: Dict[str,int], mean, std, use_img_uniform=False, augment=False):
        df = pd.read_csv(csv_path)
        self.df = df.copy()

        # choose column
        col = "img_path_uniform" if use_img_uniform and "img_path_uniform" in df.columns else "img_path"
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
        self.size = 224  # DINOv2 ViT/B14 default is 224
        mean = IMAGENET_DEFAULT_MEAN if mean is None else mean
        std  = IMAGENET_DEFAULT_STD  if std  is None else std
        
        base = [
            transforms.ToTensor(),                        # HWC [0..255] -> CHW [0..1]
            transforms.Lambda(lambda x: x.expand(3, *x.shape[1:]) if x.shape[0]==1 else x),
            transforms.Normalize(mean, std),
        ]
        if augment:
            aug = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply([transforms.RandomRotation(10, fill=0)], p=0.25),
                # (avoid crops: nucleus can be near edges after zoom)
            ]
            self.tx = transforms.Compose(aug + base)
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
