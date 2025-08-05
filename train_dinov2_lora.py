import argparse
import logging
import os
import datetime
import numpy as np
import torch
from datasets import load_from_disk, DatasetDict
from torchvision.transforms import v2
from transformers import (
    AutoImageProcessor,
    Dinov2ForImageClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
import evaluate
from sklearn.metrics import f1_score
import pandas as pd
from peft import LoraConfig, get_peft_model, TaskType
import json

# ---------- CSV Logger ----------
class CSVLoggerCallback(TrainerCallback):
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.columns = ["step", "epoch", "eval_loss", "eval_accuracy", "eval_macro_f1", "eval_micro_f1"]
        with open(self.csv_path, "w") as f:
            f.write(",".join(self.columns) + "\n")

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        row = [
            str(state.global_step),
            str(state.epoch),
            str(metrics.get("eval_loss", "")),
            str(metrics.get("eval_accuracy", "")),
            str(metrics.get("eval_macro_f1", "")),
            str(metrics.get("eval_micro_f1", "")),
        ]
        with open(self.csv_path, "a") as f:
            f.write(",".join(row) + "\n")

def print_trainable_parameters(model):
    """Print the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Trainable params: {trainable_params:,} || All params: {all_param:,} || "
          f"Trainable%: {100 * trainable_params / all_param:.4f}%")
    return trainable_params, all_param

# ---------- Arg Parser ----------
def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for DINOv2")
    parser.add_argument("--dataset_path", type=str, default="/home/tcvcs/Projects/datasets/CellImageNet")
    parser.add_argument("--output_dir", type=str, default="/home/tcvcs/Projects/CellPT/outputs/dinov2_lora)
    parser.add_argument("--wandb_project", type=str, default="dinov2-cell-adaptation")
    parser.add_argument("--model_name", type=str, default="facebook/dinov2-base")
    parser.add_argument("--num_labels", type=int, default=37)
    
    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=2000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=2000)
    
    # Optimization
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

# ---------- Main ----------
def main():
    args = parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup W&B
    os.environ["WANDB_PROJECT"] = args.wandb_project

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_output_dir = os.path.join(args.output_dir, f"dinov2_lora_r{args.lora_r}_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(run_output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}")
    raw_datasets = load_from_disk(args.dataset_path)
    assert isinstance(raw_datasets, DatasetDict)
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(raw_datasets['train'])}")
    print(f"  Validation: {len(raw_datasets['validation'])}")
    print(f"  Test: {len(raw_datasets['test'])}")

    # ---------- Processor ----------
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    print(f"Processor image size: {processor.size}")

    # Microscopy-friendly augmentations (your excellent choice!)
    train_tf = v2.Compose([
        v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(180),
        v2.ColorJitter(brightness=0.1, contrast=0.1),
        v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        v2.RandomGrayscale(p=0.05),
    ])
    
    def transform_train(examples):
        pixel_values = [
            processor(train_tf(img), return_tensors="pt")["pixel_values"].squeeze(0)
            for img in examples["image"]
        ]
        return {"pixel_values": pixel_values}

    def transform_val(examples):
        pixel_values = [
            processor(img, return_tensors="pt")["pixel_values"].squeeze(0)
            for img in examples["image"]
        ]
        return {"pixel_values": pixel_values}

    # Process datasets
    print("Processing datasets...")
    train_ds = raw_datasets["train"].rename_column("label", "labels").map(
        transform_train, batched=True, remove_columns=["image"]
    )
    val_ds = raw_datasets["validation"].rename_column("label", "labels").map(
        transform_val, batched=True, remove_columns=["image"]
    )
    test_ds = raw_datasets["test"].rename_column("label", "labels").map(
        transform_val, batched=True, remove_columns=["image"]
    )

    # ---------- Model ----------
    print(f"Loading model: {args.model_name}")
    model = Dinov2ForImageClassification.from_pretrained(
        args.model_name,
        num_labels=args.num_labels,
        ignore_mismatched_sizes=True
    )

    print("Original model parameters:")
    print_trainable_parameters(model)

    # Freeze all original params
    for param in model.parameters():
        param.requires_grad = False

    # LoRA config — applied to attention projections (your correct approach!)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],  # Correct DINOv2 names
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION  # Better choice than IMAGE_CLASSIFICATION
    )

    print(f"\nApplying LoRA with r={args.lora_r}, alpha={args.lora_alpha}")
    model = get_peft_model(model, lora_config)

    # Unfreeze classifier (critical!)
    for param in model.classifier.parameters():
        param.requires_grad = True

    print("After LoRA:")
    print_trainable_parameters(model)

    # ---------- Metrics ----------
    acc_metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=1)
        labels = eval_pred.label_ids
        acc = acc_metric.compute(predictions=preds, references=labels)["accuracy"]
        macro_f1 = f1_score(labels, preds, average="macro")
        micro_f1 = f1_score(labels, preds, average="micro")
        return {
            "accuracy": acc, 
            "macro_f1": macro_f1,
            "micro_f1": micro_f1
        }

    # ---------- Training Args ----------
    training_args = TrainingArguments(
        output_dir=run_output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=3,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to="wandb",
        run_name=f"dinov2_lora_r{args.lora_r}_{timestamp}",
        remove_unused_columns=False,
        seed=args.seed,
    )

    csv_logger = CSVLoggerCallback(csv_path=os.path.join(run_output_dir, "metrics_log.csv"))

    # ---------- Trainer ----------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[csv_logger]
    )

    print(f"\nStarting training...")
    print(f"Effective batch size: {args.train_batch_size * args.gradient_accumulation_steps}")
    
    trainer.train()

    # ---------- Test Evaluation ----------
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    
    print("\n" + "="*50)
    print("FINAL TEST RESULTS")
    print("="*50)
    for metric, value in test_metrics.items():
        if metric.startswith("test_"):
            print(f"{metric}: {value:.4f}")

    # Save test results
    with open(os.path.join(run_output_dir, "test_results.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    # ---------- Model Saving (Your excellent approach!) ----------
    print("\nSaving models...")
    
    # 1. Save LoRA adapters only (~50MB)
    lora_dir = os.path.join(run_output_dir, "lora_only")
    model.save_pretrained(lora_dir)
    print(f"LoRA adapters saved to: {lora_dir}")

    # 2. Merge LoRA with backbone and save complete model
    print("Merging LoRA with backbone...")
    merged_model = model.merge_and_unload()
    merged_dir = os.path.join(run_output_dir, "merged_model")
    merged_model.save_pretrained(merged_dir)
    processor.save_pretrained(merged_dir)
    print(f"Merged model saved to: {merged_dir}")

    # 3. Save backbone only for reuse (Your smart addition!)
    backbone_dir = os.path.join(run_output_dir, "backbone_only")
    os.makedirs(backbone_dir, exist_ok=True)
    torch.save(merged_model.dinov2.state_dict(), os.path.join(backbone_dir, "pytorch_model.bin"))
    
    # Save backbone config
    backbone_config = merged_model.dinov2.config
    backbone_config.save_pretrained(backbone_dir)
    print(f"Domain-adapted backbone saved to: {backbone_dir}")

    print(f"\nTraining complete! All outputs saved to: {run_output_dir}")

if __name__ == "__main__":
    main()