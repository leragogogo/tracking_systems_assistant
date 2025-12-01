import argparse
from pathlib import Path
from utils.clean_text import clean_body_text
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate


def generate_titles(
    model,
    tokenizer,
    bodies,
    device,
    max_source_length: int = 256,
    max_target_length: int = 32,
    num_beams: int = 4,
):
    model.eval()
    all_titles = []
    total = len(bodies)

    with torch.inference_mode():
        for i, body in enumerate(bodies):
            if i % 50 == 0:
                print(f"Generating {i}/{total}...")

            input_text = "summarize: " + body
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_source_length,
            ).to(device)

            outputs = model.generate(
                **inputs,
                max_length=max_target_length,
                num_beams=num_beams,
                early_stopping=True,
            )

            pred_title = tokenizer.decode(
                outputs[0], skip_special_tokens=True
            ).strip()
            all_titles.append(pred_title)

    return all_titles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of test examples to evaluate on.")
    args = parser.parse_args()

    data_path = Path("datasets/embold_train.json")
    model_dir = Path("model_artifacts/summarizer")

    # Load datsset
    df = pd.read_json(data_path)
    df = df[["title", "body"]].dropna()

    # Clean bodies experiment
    df["body_clean"] = df["body"].astype(str).apply(clean_body_text)

    # Drop rows where cleaning killed all content experiment
    df = df[df["body_clean"].str.len() > 0].reset_index(drop=True)

    _, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    _, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print("Full test size:", len(test_df))

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)

    pred_titles = generate_titles(
        model=model,
        tokenizer=tokenizer,
        bodies=test_df["body_clean"].tolist(),
        device=device,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )

    references = test_df["title"].tolist()

    bleu_metric = evaluate.load("sacrebleu")
    rouge_metric = evaluate.load("rouge")

    bleu_result = bleu_metric.compute(
        predictions=pred_titles,
        references=[[ref] for ref in references],
    )
    rouge_result = rouge_metric.compute(
        predictions=pred_titles,
        references=references,
        use_stemmer=True,
    )

    print("\n=== BLEU & ROUGE metrics on test set ===\n")
    print(f"BLEU (sacrebleu): {bleu_result['score']:.2f}")
    print(f"ROUGE-1: {rouge_result['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_result['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_result['rougeL']:.4f}")
    print(f"ROUGE-Lsum: {rouge_result.get('rougeLsum', 0.0):.4f}")


if __name__ == "__main__":
    main()

