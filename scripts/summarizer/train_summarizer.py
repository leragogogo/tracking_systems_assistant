import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.clean_text import clean_body_text
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Train batch size per device.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        help="Eval batch size per device.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=256,
        help="Max token length for input (body).",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=32,
        help="Max token length for target (title).",
    )
    args = parser.parse_args()

    data_path = Path("datasets/embold_train.json")
    model_name = "t5-small"
    output_dir = Path("model_artifacts/summarizer")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = pd.read_json(data_path)
    df = df[["title", "body"]].dropna()

    # Clean bodies experiment
    df["body_clean"] = df["body"].astype(str).apply(clean_body_text)

    # Drop rows where cleaning killed all content experiment
    df = df[df["body_clean"].str.len() > 0].reset_index(drop=True)


    # Split: 80% train, 10% val, 10% test (all stratified)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
    )
    val_df, _ = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
    )

    # Convert to HF Datasets
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Preprocessing: body -> input, title -> labels
    def preprocess_function(batch):
        inputs = ["summarize: " + b for b in batch["body_clean"]]
        model_inputs = tokenizer(
            inputs,
            max_length=args.max_source_length,
            padding="max_length",
            truncation=True,
        )

        # Tokenize titles
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["title"],
                max_length=args.max_target_length,
                padding="max_length",
                truncation=True,
            )

        labels_ids = labels["input_ids"]
        # Replace pad token id with -100 so loss ignores them
        labels_ids = [
            [
                (token_id if token_id != tokenizer.pad_token_id else -100)
                for token_id in seq
            ]
            for seq in labels_ids
        ]
        model_inputs["labels"] = labels_ids
        return model_inputs

    train_ds = train_ds.map(preprocess_function, batched=True)
    val_ds = val_ds.map(preprocess_function, batched=True)

    cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=cols)
    val_ds.set_format(type="torch", columns=cols)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=5e-5,
        weight_decay=0.01,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train and save
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"Summarizer model saved to: {output_dir}")


if __name__ == "__main__":
    main()
