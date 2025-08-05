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

# ---------- CSV Logger ----------
class CSVLoggerCallback(TrainerCallback):
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.columns = ["step", "eval_accuracy", "eval_macro_f1"]
        with open(self.csv_path, "w") as f:
            f.write(",".join(self.columns) + "\n")

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        row = [
            str(state.global_step),
            str(metrics.get("eval_accuracy", "")),
            str(metrics.get("eval_macro_f1", "")),
        ]
        with open(self.csv_path, "a") as f:
            f.write(",".join(row) + "\n")

# ---------- Arg Parser ----------
def parse_args():
    parser = argparse.ArgumentParser(description="Partial-unfreeze fine-tuning for DINOv2")
    parser.add_argument("--dataset_path", type=str, "/home/tcvcs/Projects/datasets/CellImageNet")
    parser.add_argument("--output_dir", type=str, default="/home/tcvcs/Projects/CellPT/outputs/dinov2_partial_unfreeze)
    parser.add_argument("--wandb_project", type=str, required=True)
    parser.add_argument("--num_labels", type=int, default=37)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=2000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--unfreeze_blocks", type=int, default=2,
                        help="Number of final transformer blocks to unfreeze")
    return parser.parse_args()

# ---------- Main ----------
def main():
    args = parse_args()

    os.environ["WANDB_PROJECT"] = args.wandb_project

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_output_dir = os.path.join(args.output_dir, f"dinov2_partial_unfreeze_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    raw_datasets = load_from_disk(args.dataset_path)
    assert isinstance(raw_datasets, DatasetDict)

    # ---------- Processor ----------
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    # Microscopy-friendly augmentations
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

    train_ds = raw_datasets["train"].rename_column("label", "labels").map(transform_train, batched=True)
    val_ds = raw_datasets["validation"].rename_column("label", "labels").map(transform_val, batched=True)
    test_ds = raw_datasets["test"].rename_column("label", "labels").map(transform_val, batched=True)

    # ---------- Model ----------
    model = Dinov2ForImageClassification.from_pretrained(
        "facebook/dinov2-base",
        num_labels=args.num_labels,
        ignore_mismatched_sizes=True
    )

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last N transformer blocks + classifier
    total_blocks = len(model.dinov2.encoder.layers)
    for i in range(total_blocks - args.unfreeze_blocks, total_blocks):
        for param in model.dinov2.encoder.layers[i].parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    # ---------- Metrics ----------
    acc_metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=1)
        labels = eval_pred.label_ids
        acc = acc_metric.compute(predictions=preds, references=labels)["accuracy"]
        macro_f1 = f1_score(labels, preds, average="macro")
        return {"accuracy": acc, "macro_f1": macro_f1}

    # ---------- Training Args ----------
    training_args = TrainingArguments(
        output_dir=run_output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        report_to="wandb",
        run_name=f"dinov2_partial_unfreeze_{timestamp}",
        remove_unused_columns=False
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

    trainer.train()

    # ---------- Test ----------
    test_metrics = trainer.evaluate(test_ds)
    print("Test metrics:", test_metrics)

    # Save backbone-only weights
    backbone_dir = os.path.join(run_output_dir, "backbone_only")
    os.makedirs(backbone_dir, exist_ok=True)
    torch.save(model.dinov2.state_dict(), os.path.join(backbone_dir, "pytorch_model.bin"))

if __name__ == "__main__":
    main()
