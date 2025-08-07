import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

from model.dinov2_model import get_model

from data.dataset import load_arrow_datasets
from model.model_lora import apply_lora_to_dinov2
from utils.save_load import save_full_model, save_lora_only
from utils.logger import CSVLogger



# ----- Transforms -----
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ----- Evaluation -----
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    micro_f1 = f1_score(all_labels, all_preds, average="micro")

    return acc, macro_f1, micro_f1


# ----- Training -----
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for imgs, labels in tqdm(dataloader, desc="Training"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


# ----- Main -----
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = CSVLogger("logs/train_lora_metrics.csv", header=["epoch", "train_loss", "accuracy", "macro_f1", "micro_f1"])

    # Load datasets
    train_ds, val_ds, _ = load_arrow_datasets(
        data_dir="/home/tcvcs/Projects/datasets/CellImageNet",
        transform_train=transform_train,
        transform_val=transform_val
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    # Load model
    model = get_model(num_classes=37, freeze_backbone=True).to(device)
    # Inject LoRA adapters into specified layers
    model = apply_lora_to_dinov2(model, r=8, alpha=16).to(device)

    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Optimizer and loss
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(10):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc, macro_f1, micro_f1 = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Acc: {acc:.4f} | Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f}")
        logger.log_row([epoch+1, train_loss, acc, macro_f1, micro_f1])


    save_full_model(model, "checkpoints/dinov2_lora_full.pth")
    save_lora_only(model, "checkpoints/dinov2_lora_adapters.pth")
    
    """to load later ----
    from utils.save_load import load_full_model

    model = get_model(num_classes=4, freeze_backbone=True)
    model = apply_lora_to_dinov2(model, r=8, alpha=16)
    model = load_full_model(model, "checkpoints/dinov2_lora_full.pth").to(device)"""

if __name__ == "__main__":
    main()
