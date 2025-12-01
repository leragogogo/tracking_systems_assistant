from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SUMMARIZER_ID: str = "leragogogo/github-issues-summarizer"
    CLASSIFIER_ID: str = "leragogogo/github-issues-classifier"

    DEVICE: str = "cpu"

settings = Settings()
