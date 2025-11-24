import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch import nn

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


# Trainer subclass with weighted loss
class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
            self,
            model,
            inputs,
            return_outputs: bool = False,
            num_items_in_batch: int | None = None,  # <- new param to match Trainer API
    ):
        # We don't actually use num_items_in_batch, but we must accept it
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        weight = self.class_weights.to(logits.device)

        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# Tokenization function
def tokenize_fn(example, tokenizer, max_length: int = 256):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="datasets/embold_train.json",
        help="Path to dataset (e.g. Train.json from Kaggle)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Train & eval batch size",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Max sequence length for tokenizer",
    )
    args = parser.parse_args()

    model_name = "distilbert-base-uncased"
    data_path = Path(args.data_path)
    output_dir = Path("model_artifacts/classifier")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = pd.read_json(data_path)

    # Build a single 'text' column
    df["text"] = df["title"].fillna("") + " " + df["body"].fillna("")

    # Split: 80% train, 10% val, 10% test (all stratified)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )

    val_df, _ = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=42,
    )

    # Compute class weights from train split
    class_labels = np.unique(train_df["label"].values)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=class_labels,
        y=train_df["label"].values,
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    print("Class labels:", class_labels)
    print("Class weights:", class_weights)

    # Convert to HF Datasets
    train_ds = Dataset.from_pandas(train_df[["text", "label"]])
    val_ds = Dataset.from_pandas(val_df[["text", "label"]])

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
    )

    # Tokenize datasets
    def _tokenize(batch):
        return tokenize_fn(batch, tokenizer, max_length=args.max_length)

    train_ds = train_ds.map(_tokenize, batched=True)
    val_ds = val_ds.map(_tokenize, batched=True)

    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch", 
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=5e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
    )

    # Metrics on validation
    def compute_metrics(pred):
        logits = pred.predictions
        labels = pred.label_ids
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average="macro")
        return {"accuracy": acc, "f1_macro": f1_macro}

    # Create Trainer with weighted loss
    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train and save
    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
