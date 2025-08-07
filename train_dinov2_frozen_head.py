import argparse
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
    TrainingArguments
)
import evaluate
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Frozen-backbone linear probe for DINOv2")
    parser.add_argument("--dataset_path", type=str, default="/home/tcvcs/Projects/datasets/CellImageNet")
    parser.add_argument("--output_dir", type=str, default="/home/tcvcs/Projects/CellPT/outputs/dinov2_frozen_head")
    parser.add_argument("--num_labels", type=int, default=37)
    parser.add_argument("--learning_rate", type=float, default=1e-3)  # higher LR for head-only training
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=100)
    return parser.parse_args()

def main():
    args = parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_output_dir = os.path.join(args.output_dir, f"dinov2_frozen_head_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    raw_datasets = load_from_disk(args.dataset_path)
    assert isinstance(raw_datasets, DatasetDict)

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    train_tf = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(180),
    ])

    # On‑the‑fly collation instead of .map()
    def collate_fn_train(batch):
        pixel_values = [
            processor(train_tf(item["image"]), return_tensors="pt")["pixel_values"].squeeze(0)
            for item in batch
        ]
        labels = [item["label"] for item in batch]
        return {
            "pixel_values": torch.stack(pixel_values),
            "labels": torch.tensor(labels)
        }

    def collate_fn_eval(batch):
        pixel_values = [
            processor(item["image"], return_tensors="pt")["pixel_values"].squeeze(0)
            for item in batch
        ]
        labels = [item["label"] for item in batch]
        return {
            "pixel_values": torch.stack(pixel_values),
            "labels": torch.tensor(labels)
        }

    model = Dinov2ForImageClassification.from_pretrained(
        "facebook/dinov2-base",
        num_labels=args.num_labels,
        ignore_mismatched_sizes=True
    )
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    acc_metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=1)
        labels = eval_pred.label_ids
        acc = acc_metric.compute(predictions=preds, references=labels)["accuracy"]
        macro_f1 = f1_score(labels, preds, average="macro")
        return {"accuracy": acc, "macro_f1": macro_f1}

    training_args = TrainingArguments(
        output_dir=run_output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["validation"],
        compute_metrics=compute_metrics,
        data_collator=collate_fn_train  # will be overridden for eval below
    )

    # Patch eval collator
    trainer.get_eval_dataloader = lambda eval_dataset=None: DataLoader(
        eval_dataset if eval_dataset is not None else trainer.eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn_eval
    )

    trainer.train()

    test_metrics = trainer.evaluate(
        DataLoader(raw_datasets["test"], batch_size=args.eval_batch_size, collate_fn=collate_fn_eval)
    )
    print("Test metrics:", test_metrics)

if __name__ == "__main__":
    main()