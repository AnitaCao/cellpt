from datasets import load_from_disk
from torch.utils.data import Dataset
from PIL import Image
import torch

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
