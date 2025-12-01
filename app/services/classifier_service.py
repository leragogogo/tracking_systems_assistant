import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.core.config import settings

class ClassifierService:
    """
    This service is responsible for:
        - loading the fine-tuned DistilBERT classification model
        - preprocessing input issue descriptions (tokenization)
        - returning both the predicted label and probability scores
    
    The model predicts three classes:
        0: "bug"
        1: "feature"
        2: "question"

    After initialization, the model is kept in memory and reused for
    every incoming request, ensuring fast inference.
    """
    def __init__(self):
        self.device = torch.device(settings.DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.CLASSIFIER_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(settings.CLASSIFIER_ID).to(self.device)
        self.model.eval()

        self.id_to_label = {
            0: "bug",
            1: "feature",
            2: "question"
        }

    def predict_type(self, body: str):
        """
        Predict the issue type (bug, feature, question) based on the text body.

        Args:
            body (str): Issue description provided by the user.

        Returns:
            tuple:
                - predicted_label (str)
                - probabilities (dict with float values for each class)
        """
        inputs = self.tokenizer(body, return_tensors="pt", padding=True, truncation=True).to(self.device)

        with torch.inference_mode():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        label_id = int(probs.argmax())
        label = self.id_to_label[label_id]

        return label, {
            "bug": float(probs[0]),
            "feature": float(probs[1]),
            "question": float(probs[2]),
        }
