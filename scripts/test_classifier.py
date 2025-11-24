import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
)


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
        help="Path to the same dataset used in training (e.g. Train.json)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Max sequence length for tokenizer",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    model_id = "leragogogo/github-issues-classifier"

    # Load dataset
    df = pd.read_json(data_path)

    df["text"] = df["title"].fillna("") + " " + df["body"].fillna("")

    # Recreate the same 80/10/10 split
    _, temp_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )

    _, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=42,
    )

    print("Evaluation on TEST split with shape:", test_df.shape)

    # Convert test_df to HF Dataset
    test_ds = Dataset.from_pandas(test_df[["text", "label"]])

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    def _tokenize(batch):
        return tokenize_fn(batch, tokenizer, max_length=args.max_length)

    test_ds = test_ds.map(_tokenize, batched=True)

    test_ds = test_ds.rename_column("label", "labels")
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
    )
    
    print("Running prediction on test set...")
    preds_output = trainer.predict(test_ds)

    logits = preds_output.predictions
    labels = preds_output.label_ids
    preds = np.argmax(logits, axis=-1)

    # Display metrics
    print("\nClassification report:")
    print(classification_report(labels, preds, digits=4))

    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    main()
