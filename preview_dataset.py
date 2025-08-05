from datasets import load_from_disk

import matplotlib.pyplot as plt
%matplotlib inline
import random

# Path to your dataset split (train, validation, or test)
dataset_path = "/home/tcvcs/Projects/datasets/CellImageNet/train"
dataset = load_from_disk(dataset_path)

# Get label names if available
label_names = dataset.features["label"].names if hasattr(dataset.features["label"], "names") else None

# Number of images to show per class
images_per_class = 5

# Create a mapping from label -> list of indices
label_to_indices = {}
for idx, label_id in enumerate(dataset["label"]):
    label_to_indices.setdefault(label_id, []).append(idx)

# Display grid for each class
for label_id, indices in label_to_indices.items():
    label_str = label_names[label_id] if label_names else str(label_id)
    sample_indices = random.sample(indices, min(images_per_class, len(indices)))

    fig, axes = plt.subplots(1, len(sample_indices), figsize=(15, 3))
    fig.suptitle(f"Class: {label_str}", fontsize=14)

    for ax, idx in zip(axes, sample_indices):
        img = dataset[idx]["image"]  # PIL.Image
        ax.imshow(img)
        ax.axis("off")

    plt.show()
    plt.close(fig)


from datasets import load_from_disk
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
dataset_path = "/home/tcvcs/Projects/datasets/CellImageNet"
ds = load_from_disk(dataset_path)

# Work with the train split
labels = ds["train"]["label"]  # or "labels" if you already renamed
df = pd.DataFrame({"label": labels})

# Count per class
counts = df["label"].value_counts().sort_index()

# Plot histogram
plt.figure(figsize=(12, 6))
counts.plot(kind="bar")
plt.xlabel("Class ID")
plt.ylabel("Number of samples")
plt.title("Class Distribution in Train Split")
plt.show()
