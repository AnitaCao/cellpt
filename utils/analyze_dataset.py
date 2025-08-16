import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk
from collections import Counter
from rich import print as rprint
from rich.table import Table

# --------- Config ---------
DATA_DIR = "/hpc/group/jilab/rz179/finalized_data/standard/CellImageNet"
SAVE_DIR = "logs/dataset_analysis"
os.makedirs(SAVE_DIR, exist_ok=True)

# --------- Load dataset ---------
ds = load_from_disk(DATA_DIR)
train = ds['train']
val = ds['validation']
test = ds['test'] if 'test' in ds else None

# --------- Helper: label names ---------
def get_label_names(dataset):
    if 'label' in dataset.features and hasattr(dataset.features['label'], 'names'):
        return dataset.features['label'].names
    return [str(i) for i in range(max(dataset['label']) + 1)]

label_names = get_label_names(train)
num_classes = len(label_names)

# --------- Distribution counts ---------
def count_labels(dataset):
    return Counter(dataset['label'])

train_counts = count_labels(train)
val_counts = count_labels(val)
test_counts = count_labels(test) if test else {}

# --------- Table Summary ---------
def print_table(counts, name):
    table = Table(title=f"{name} Class Distribution")
    table.add_column("Class ID", justify="right")
    table.add_column("Class Name")
    table.add_column("Count", justify="right")
    for i in range(num_classes):
        count = counts.get(i, 0)
        label = label_names[i] if i < len(label_names) else str(i)
        table.add_row(str(i), label, str(count))
    rprint(table)

print_table(train_counts, "Train")
print_table(val_counts, "Validation")
if test:
    print_table(test_counts, "Test")

# --------- Save JSON summary ---------
summary = {
    'num_classes': num_classes,
    'train_distribution': dict(train_counts),
    'val_distribution': dict(val_counts),
    'test_distribution': dict(test_counts) if test else {},
    'label_names': label_names
}

with open(os.path.join(SAVE_DIR, "class_distribution_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved JSON summary to: {SAVE_DIR}/class_distribution_summary.json")

# --------- Plot class distribution ---------
def plot_distribution(counts, name):
    fig, ax = plt.subplots(figsize=(12, 5))
    xs = np.arange(num_classes)
    ys = [counts.get(i, 0) for i in xs]
    ax.bar(xs, ys)
    ax.set_title(f"{name} Set Class Distribution")
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Sample Count")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(i) for i in xs], rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{name.lower()}_distribution.png"))
    plt.close()

plot_distribution(train_counts, "Train")
plot_distribution(val_counts, "Validation")
if test:
    plot_distribution(test_counts, "Test")

print("Plots saved in:", SAVE_DIR)
