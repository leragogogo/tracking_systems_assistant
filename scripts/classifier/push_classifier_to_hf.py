from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    # Directory with trained classifier
    local_model_dir = Path("model_artifacts/classifier")

    hf_repo_name = "github-issues-classifier"  
    hf_namespace = "leragogogo"           

    repo_id = f"{hf_namespace}/{hf_repo_name}"

    print(f"Loading model from {local_model_dir}...")
    model = AutoModelForSequenceClassification.from_pretrained(local_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)

    print(f"Pushing model to Hugging Face Hub as {repo_id} ...")
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)

    print("Done! Model is now on the Hub.")


if __name__ == "__main__":
    main()
