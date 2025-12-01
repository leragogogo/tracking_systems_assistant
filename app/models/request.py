from pydantic import BaseModel

class IssueRequest(BaseModel):
    body: str