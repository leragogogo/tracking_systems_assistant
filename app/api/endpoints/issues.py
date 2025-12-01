from fastapi import APIRouter
from app.models.request import IssueRequest
from app.models.response import IssueAssistResponse
from app.services.models_manager import models_manager
from utils.clean_text import clean_body_text

router = APIRouter()

@router.post("/assist", response_model=IssueAssistResponse)
async def assist_issue(req: IssueRequest):
    body = req.body
    cleaned_body = clean_body_text(body)

    # Summarization
    title = models_manager.summarizer.generate_title(cleaned_body)

    # Classification
    label, label_probs = models_manager.classifier.predict_type(body)

    return IssueAssistResponse(
        title=title,
        label=label,
        label_probs=label_probs,
    )
