import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report 

from model.dinov2_model import get_model
from model.model_lora import apply_lora_to_dinov2
from data.dataset import load_arrow_datasets
from utils.save_load import save_full_model, save_lora_only
from utils.logger import CSVLogger


# ----- Transforms -----
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-180, 180)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    transforms.RandomAutocontrast(p=0.2),
    transforms.ToTensor(),
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ----- Evaluation -----
def evaluate(model, epoch, dataloader, device, criterion=None):
    model.eval()
    all_preds, all_labels = [], []
    total_val_loss = 0.0

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            if criterion:
                total_val_loss += criterion(outputs, labels).item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    micro_f1 = f1_score(all_labels, all_preds, average="micro")
    avg_val_loss = total_val_loss / len(dataloader) if criterion else None
    
    if epoch % 2 == 0:  # Every 2 epochs
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        class_f1s = {i: report[str(i)]['f1-score'] for i in range(37) if str(i) in report}
        
        # Show bottom 5 performing classes
        worst_classes = sorted(class_f1s.items(), key=lambda x: x[1])[:5]
        print(f"Bottom 5 classes (F1): {worst_classes}")
        
        # Show overall worst and best
        if class_f1s:
            min_f1 = min(class_f1s.values())
            max_f1 = max(class_f1s.values())
            print(f"Class performance range: {min_f1:.3f} - {max_f1:.3f}")
            
        #print 

    return acc, macro_f1, micro_f1, avg_val_loss

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

    logger = CSVLogger("logs/train_lora_metrics.csv", header=["epoch", "train_loss", "val_loss", "accuracy", "macro_f1", "micro_f1"])

    # Load datasets
    train_ds, val_ds, _ = load_arrow_datasets(
        data_dir="/hpc/group/jilab/rz179/finalized_data/standard/CellImageNet",
        transform_train=transform_train,
        transform_val=transform_val
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    # Load model
    model = get_model(num_classes=37, freeze_backbone=True).to(device)
    model = apply_lora_to_dinov2(model, r=16, alpha=16).to(device)

    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Optimizer, scheduler, loss
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-4,
        weight_decay=1e-2
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optional: early stopping
    best_val_loss = float("inf")
    early_stop_patience = 5
    no_improve = 0

    for epoch in range(10):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc, macro_f1, micro_f1, val_loss = evaluate(model, epoch, val_loader, device, criterion)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {acc:.4f} | Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f}")
        logger.log_row([epoch+1, train_loss, val_loss, acc, macro_f1, micro_f1])

        scheduler.step()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            save_full_model(model, "checkpoints/dinov2_lora_best_full.pth")
            save_lora_only(model, "checkpoints/dinov2_lora_best_lora_only.pth")
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

if __name__ == "__main__":
    main()
