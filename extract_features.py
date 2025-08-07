import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModel, AutoImageProcessor

@torch.no_grad()
def extract_and_save_features(model_name_or_path, dataset_path, output_dir, batch_size=64, device="cuda"):
    # Load model & processor
    print(f"Loading model: {model_name_or_path}")
    model = AutoModel.from_pretrained(model_name_or_path).to(device)
    processor = AutoImageProcessor.from_pretrained(model_name_or_path)
    model.eval()

    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    label_names = dataset.features["label"].names if hasattr(dataset.features["label"], "names") else None

    os.makedirs(output_dir, exist_ok=True)

    all_features = []
    all_labels = []

    # Process in batches
    for start_idx in tqdm(range(0, len(dataset), batch_size), desc="Extracting features"):
        batch = dataset[start_idx:start_idx + batch_size]
        images = batch["image"]
        labels = batch["label"]

        inputs = processor(images=images, return_tensors="pt").to(device)
        outputs = model(**inputs)

        # Get pooled features
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feats = outputs.pooler_output
        else:
            feats = outputs.last_hidden_state[:, 0]  # CLS token

        # Normalize each feature vector
        feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-10)

        all_features.extend(feats.cpu().numpy())
        all_labels.extend(labels)

    # Save per-class .npz files
    print("Saving per-class feature files...")
    label_set = sorted(set(all_labels))
    for label in label_set:
        indices = [i for i, y in enumerate(all_labels) if y == label]
        class_feats = np.stack([all_features[i] for i in indices])
        class_labels = np.array([all_labels[i] for i in indices])
        label_str = label_names[label] if label_names else str(label)
        np.savez_compressed(os.path.join(output_dir, f"{label_str}.npz"),
                            features=class_feats, labels=class_labels)

    print(f"Saved features to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or path (e.g., openai/clip-vit-base-patch32, facebook/dinov2-base, or local fine-tuned checkpoint)")
    parser.add_argument("--dataset_path", default="/home/tcvcs/Projects/datasets/CellImageNet/test", type=str, help="Path to dataset split folder (train/validation/test)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save extracted features")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for feature extraction")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extract_and_save_features(args.model, args.dataset_path, args.output_dir, args.batch_size, device)
