from app.services.lstm import LSTMModel
from app.services.gru import GRUModel
from app.services.transformers import TransformersModel

class ModelManager:
    def __init__(self):
        self.models = {
            "lstm": LSTMModel(),
            "gru": GRUModel(),
            "transformers": TransformersModel(),
        }

    def get_model(self, name: str):
        return self.models.get(name)

model_manager = ModelManager()