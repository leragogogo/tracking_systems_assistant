# Issue Assistant API
AI-powered backend service that generates issue titles and predicts issue types from issue body(description).
Designed for integration with GitHub, Jira, YouTrack, and other issue-tracking systems.

It provides a FastAPI-based API that wraps two fine-tuned transformer models:
- T5-small for title generation
- DistilBERT for issue type classification (bug, feature, question)
Both models are loaded once from Hugging Face at startup and served through an /api/issues/analyze endpoint.

Classification model is available at: **[Hugging Face](https://huggingface.co/leragogogo/github-issues-classifier)** <br>
Summarization model is available at: **[Hugging Face](https://huggingface.co/leragogogo/github-issues-summarizer)**

## Prerequisites
- Python 3.10+
- Internet access on first run (to download models from Hugging Face)

## Setup

### Clone the repository
```python
git clone https://github.com/leragogogo/tracking_systems_assistant.git
cd tracking_systems_assistant
```

### Create a virtual environment
```python
python3 -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

### Install dependencies
```python
pip install -r requirements.txt
```

## Running the API
Start the FastAPI app via Uvicorn:
```python
uvicorn app.main:app --reload
```

Swagger UI: http://127.0.0.1:8000/docs if you want to easily try it out

## Using the endpoint

Send a POST request to:
```python
POST /api/issues/assist
```

Example JSON body:
```python
{
  "body": "When I click the save button, the application crashes with a TypeError..."
}
```

Example response:
```python
{
  "title": "Fix crash when clicking save button",
  "issue_type": "bug",
  "probabilities": {
    "bug": 0.87,
    "feature": 0.09,
    "question": 0.04
  }
}
```

