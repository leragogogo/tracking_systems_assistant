from pydantic import BaseModel
from typing import Dict

class IssueAssistResponse(BaseModel):
    title: str
    label: str
    label_probs: Dict[str, float]