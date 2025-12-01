import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from app.core.config import settings

class SummarizerService:
    """
    This service is responsible for:
        - loading the fine-tuned T5-small summarization model
        - preparing text as input to the model ("summarize: <text>")
        - generating a concise issue title from an issue description
    """
    def __init__(self):
        self.device = torch.device(settings.DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.SUMMARIZER_ID)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(settings.SUMMARIZER_ID).to(self.device)
        self.model.eval()

    def generate_title(self, body: str, max_len: int = 32) -> str:
        input_text = "summarize: " + body
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(self.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_length=max_len,
                num_beams=4,
                early_stopping=True,
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
