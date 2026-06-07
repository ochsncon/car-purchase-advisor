"""Train a transfer-learning vehicle brand classifier using ResNet-18.

This script uses:
- datasets.load_dataset("imagefolder") for data loading
- AutoImageProcessor for image preprocessing
- AutoModelForImageClassification for transfer learning
- transformers.Trainer for training
- Proper data augmentation and evaluation

The trained model is saved in Hugging Face format under models/car-image-classifier/
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import evaluate
import numpy as np
from datasets import load_dataset
from sklearn.metrics import classification_report
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_RAW_DIR, MODEL_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Train a transfer-learning vehicle brand classifier.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw/Cars Dataset",
        help="Folder with train/ and test/ subfolders containing class folders.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="microsoft/resnet-18",
        help="Base model identifier from Hugging Face.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/car-image-classifier",
        help="Output directory for trained model.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone and only train head.")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to Hugging Face Hub.")
    parser.add_argument("--hub_model_id", type=str, default="", help="Hub model ID for pushing.")
    return parser.parse_args()


def build_transforms(processor):
    """Build training and validation transforms based on processor config."""
    image_mean = processor.image_mean
    image_std = processor.image_std
    size_cfg = processor.size

    # Extract image size from processor config
    if isinstance(size_cfg, dict):
        size = size_cfg.get("shortest_edge") or size_cfg.get("height") or size_cfg.get("width") or 224
    else:
        size = int(size_cfg) if size_cfg else 224

    from torchvision.transforms import (
        CenterCrop,
        ColorJitter,
        Compose,
        Normalize,
        RandomHorizontalFlip,
        RandomResizedCrop,
        RandomRotation,
        Resize,
        ToTensor,
    )

    train_tfm = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            RandomRotation(15),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ToTensor(),
            Normalize(mean=image_mean, std=image_std),
        ]
    )

    val_tfm = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            Normalize(mean=image_mean, std=image_std),
        ]
    )

    return train_tfm, val_tfm


def main():
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    print(f"Loading dataset from {data_dir}...")
    ds = load_dataset("imagefolder", data_dir=str(data_dir))

    # Ensure we have train and test (or validation)
    if "train" not in ds:
        raise ValueError("Dataset must have a 'train' split (in train/ folder).")

    if "test" not in ds:
        if "validation" in ds:
            ds["test"] = ds["validation"]
        else:
            # Create test split if only train exists
            split = ds["train"].train_test_split(test_size=0.2, seed=42)
            ds["train"] = split["train"]
            ds["test"] = split["test"]

    # Get label mapping
    labels = ds["train"].features["label"].names
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}

    print(f"Classes: {labels}")
    print(f"Number of classes: {len(labels)}")
    print(f"Training samples: {len(ds['train'])}")
    print(f"Test samples: {len(ds['test'])}")

    # Load processor and model
    print(f"Loading base model: {args.base_model}")
    processor = AutoImageProcessor.from_pretrained(args.base_model)
    model = AutoModelForImageClassification.from_pretrained(
        args.base_model,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # Optionally freeze backbone
    if args.freeze_backbone:
        print("Freezing backbone, only training head...")
        trainable_heads = ("classifier", "score", "fc", "heads", "head")
        for name, param in model.named_parameters():
            if not any(head in name for head in trainable_heads):
                param.requires_grad = False

    # Build transforms
    train_tfm, val_tfm = build_transforms(processor)

    def transform_train(batch):
        batch["pixel_values"] = [train_tfm(img.convert("RGB")) for img in batch["image"]]
        return batch

    def transform_val(batch):
        batch["pixel_values"] = [val_tfm(img.convert("RGB")) for img in batch["image"]]
        return batch

    ds["train"].set_transform(transform_train)
    ds["test"].set_transform(transform_val)

    def collate_fn(batch):
        import torch

        return {
            "pixel_values": torch.stack([example["pixel_values"] for example in batch]),
            "labels": torch.tensor([example["label"] for example in batch]),
        }

    # Metrics
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels_ = eval_pred
        predictions = np.argmax(logits, axis=1)
        return metric.compute(predictions=predictions, references=labels_)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        label_smoothing_factor=args.label_smoothing,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.hub_model_id else None,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Evaluate
    print("Evaluating...")
    metrics = trainer.evaluate()
    print(f"Test accuracy: {metrics.get('eval_accuracy', 0):.4f}")

    # Save model and processor
    print(f"Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # Generate detailed metrics
    predictions = trainer.predict(ds["test"])
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids

    report = classification_report(
        true_labels,
        pred_labels,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    # Save metadata
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "number_of_classes": len(labels),
        "class_names": labels,
        "train_image_count": len(ds["train"]),
        "test_image_count": len(ds["test"]),
        "accuracy": float(metrics.get("eval_accuracy", 0)),
        "classification_report": report,
        "note": "This model can only predict one of the trained vehicle brands/classes. It does not provide damage detection or technical condition assessment.",
    }

    metadata_path = Path(args.output_dir) / "vision_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {metadata_path}")
    print("\nTraining complete!")
    print(f"Model saved to {args.output_dir}")
    print(f"Test accuracy: {metrics.get('eval_accuracy', 0):.4f}")

    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()