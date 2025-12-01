from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main():
    # Directory with trained summarizer
    local_model_dir = Path("model_artifacts/summarizer")

    hf_repo_name = "github-issues-summarizer"  
    hf_namespace = "leragogogo"           

    repo_id = f"{hf_namespace}/{hf_repo_name}"

    print(f"Loading model from {local_model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(local_model_dir)

    print(f"Pushing model to Hugging Face Hub as {repo_id} ...")
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)

    print("Done! Model is now on the Hub.")


if __name__ == "__main__":
    main()
