import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report
import numpy as np

from model.dinov2_model import get_model
from data.dataset import load_arrow_datasets
from utils.save_load import save_full_model
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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ----- Evaluation -----
def evaluate(model, epoch, dataloader, device, criterion=None):
    model.eval()
    all_preds, all_labels = [], []
    total_val_loss = 0.0

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Use AMP for inference too
            with torch.cuda.amp.autocast():
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
    
    # Detailed classification report every 2 epochs
    if epoch % 2 == 0:
        print(f"\n=== DETAILED CLASSIFICATION REPORT - EPOCH {epoch} ===")
        
        # Full classification report
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        
        # Per-class metrics
        class_metrics = []
        for i in range(37):
            if str(i) in report:
                precision = report[str(i)]['precision']
                recall = report[str(i)]['recall']
                f1 = report[str(i)]['f1-score']
                support = report[str(i)]['support']
                class_metrics.append((i, precision, recall, f1, support))
        
        # Sort by F1 score (worst to best)
        class_metrics.sort(key=lambda x: x[3])
        
        print("WORST PERFORMING CLASSES:")
        print("Class | Precision | Recall | F1-Score | Support")
        print("-" * 50)
        for i, (cls, prec, rec, f1, sup) in enumerate(class_metrics[:10]):  # Bottom 10
            print(f"{cls:5d} | {prec:9.3f} | {rec:6.3f} | {f1:8.3f} | {int(sup):7d}")
        
        print("\nBEST PERFORMING CLASSES:")
        print("Class | Precision | Recall | F1-Score | Support")
        print("-" * 50)
        for i, (cls, prec, rec, f1, sup) in enumerate(class_metrics[-10:]):  # Top 10
            print(f"{cls:5d} | {prec:9.3f} | {rec:6.3f} | {f1:8.3f} | {int(sup):7d}")
        
        # Summary statistics
        f1_scores = [x[3] for x in class_metrics]
        print(f"\nCLASS PERFORMANCE SUMMARY:")
        print(f"Min F1: {min(f1_scores):.3f} | Max F1: {max(f1_scores):.3f}")
        print(f"Mean F1: {np.mean(f1_scores):.3f} | Std F1: {np.std(f1_scores):.3f}")
        print(f"Classes with F1 < 0.5: {sum(1 for f1 in f1_scores if f1 < 0.5)}/37")
        print(f"Classes with F1 > 0.8: {sum(1 for f1 in f1_scores if f1 > 0.8)}/37")
        print("=" * 60)

    return acc, macro_f1, micro_f1, avg_val_loss


# ----- Partial Unfreeze -----
def unfreeze_last_n_layers(model, n=2):
    """
    Unfreeze the last n transformer blocks of DINOv2
    """
    print(f"Attempting to unfreeze last {n} layers...")
    
    # Try to find the transformer blocks - adapt this to your model structure
    blocks = None
    try:
        # Common DINOv2 structures to try:
        if hasattr(model, 'backbone'):
            if hasattr(model.backbone, 'blocks'):
                blocks = model.backbone.blocks
            elif hasattr(model.backbone, 'model') and hasattr(model.backbone.model, 'encoder'):
                if hasattr(model.backbone.model.encoder, 'layer'):
                    blocks = model.backbone.model.encoder.layer
                elif hasattr(model.backbone.model.encoder, 'blocks'):
                    blocks = model.backbone.model.encoder.blocks
        elif hasattr(model, 'blocks'):
            blocks = model.blocks
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            blocks = model.encoder.layer
        
        if blocks is None:
            raise AttributeError("Cannot find transformer blocks")
        
        total_blocks = len(blocks)
        print(f"Found {total_blocks} transformer blocks")
        
        # Unfreeze last n blocks
        for i in range(max(0, total_blocks - n), total_blocks):
            print(f"Unfreezing block {i}")
            for param in blocks[i].parameters():
                param.requires_grad = True
                
    except Exception as e:
        print(f"Error in unfreezing: {e}")
        print("Model structure debug info:")
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # leaf modules
                total_params = sum(p.numel() for p in module.parameters())
                if total_params > 1000:  # Only show modules with significant parameters
                    print(f"  {name}: {total_params} parameters")
        
        print("\nFalling back to manual unfreezing - please check model structure")
        # Manual fallback - unfreeze parameters with specific patterns
        unfrozen_count = 0
        for name, param in model.named_parameters():
            # Try to identify later layers by name patterns
            if any(pattern in name for pattern in ['layer.10', 'layer.11', 'blocks.10', 'blocks.11', 
                                                  'encoder.10', 'encoder.11']):
                param.requires_grad = True
                unfrozen_count += 1
        
        if unfrozen_count > 0:
            print(f"Manually unfroze {unfrozen_count} parameters")
        else:
            print("Warning: Could not unfreeze any layers automatically")


# ----- Optimizer with Layer-specific Learning Rates -----
def get_optimizer_with_layer_lr(model, base_lr=5e-5, head_lr=1e-4):
    """
    Use lower learning rate for unfrozen backbone, higher for head
    """
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if any(keyword in name.lower() for keyword in ['head', 'classifier', 'fc']):
                head_params.append(param)
            else:
                backbone_params.append(param)
    
    print(f"Backbone parameters: {sum(p.numel() for p in backbone_params)}")
    print(f"Head parameters: {sum(p.numel() for p in head_params)}")
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': base_lr, 'weight_decay': 1e-2},
        {'params': head_params, 'lr': head_lr, 'weight_decay': 1e-2}
    ])
    
    return optimizer


# ----- Training with AMP -----
def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0

    for imgs, labels in tqdm(dataloader, desc="Training"):
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with autocast
        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()

    return total_loss / len(dataloader)


# ----- Main -----
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Print GPU info
    if torch.cuda.is_available():
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    logger = CSVLogger("logs/train_partial_metrics.csv", 
                      header=["epoch", "train_loss", "val_loss", "accuracy", "macro_f1", "micro_f1"])

    # Load datasets
    train_ds, val_ds, _ = load_arrow_datasets(
        data_dir="/hpc/group/jilab/rz179/finalized_data/standard/CellImageNet",
        transform_train=transform_train,
        transform_val=transform_val
    )

    # Scale batch size and workers for multi-GPU
    num_gpus = torch.cuda.device_count()
    batch_size = 32 * max(1, num_gpus)  # Scale batch size
    num_workers = min(8, torch.cuda.device_count() * 4)  # Scale workers
    
    print(f"Using batch size: {batch_size}, num_workers: {num_workers}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, 
                           num_workers=num_workers, pin_memory=True)

    # Load model with frozen backbone
    model = get_model(num_classes=37, freeze_backbone=True)
    
    # Multi-GPU setup
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(device)

    # Unfreeze last N encoder layers
    unfreeze_last_n_layers(model, n=2)  # Start with 2 layers

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Optimizer with layer-specific learning rates
    optimizer = get_optimizer_with_layer_lr(model, base_lr=5e-5, head_lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Initialize AMP scaler
    scaler = torch.cuda.amp.GradScaler()
    print("AMP (Automatic Mixed Precision) enabled for faster training")

    # Training loop with early stopping
    best_val_loss = float("inf")
    early_stop_patience = 5
    no_improve = 0

    for epoch in range(15):  # More epochs for partial training
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        acc, macro_f1, micro_f1, val_loss = evaluate(model, epoch, val_loader, device, criterion)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Acc: {acc:.4f} | Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f}")
        
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.2e}")
        
        logger.log_row([epoch+1, train_loss, val_loss, acc, macro_f1, micro_f1])

        scheduler.step()

        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            save_full_model(model, "checkpoints/dinov2_partial_best.pth")
            print(f"New best model saved (val_loss: {val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        if epoch == 5:  # Unfreeze more layers halfway through
            print("Unfreezing more layers...")
            unfreeze_last_n_layers(model, n=4)
            # Update optimizer with new parameters
            optimizer = get_optimizer_with_layer_lr(model, base_lr=3e-5, head_lr=5e-5)
            scaler = torch.cuda.amp.GradScaler()

    print("Training complete.")

if __name__ == "__main__":
    main()
