from app.services.summarizer_service import SummarizerService
from app.services.classifier_service import ClassifierService

class ModelsManager:
    """
    Central container for all ML services
    used in the application.
    """
    def __init__(self):
        self.summarizer = SummarizerService()
        self.classifier = ClassifierService()

models_manager = ModelsManager()
