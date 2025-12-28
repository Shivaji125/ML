# src/inference/predictor.py

import joblib
import pandas as pd
from pathlib import Path

from .validation import InferenceDataValidator
from src.utils.paths import get_config_path
from src.utils.config_loader import load_config


class ModelPredictor:
    """Loads trained artifacts and performs inference."""

    def __init__(self):
        self.config = load_config(get_config_path("paths_config.yaml"))

        model_dir = Path(self.config["MODEL_DIR"])
        self.preprocessor_path = model_dir / self.config["PREPROCESSOR_FILENAME"]
        self.model_path = model_dir / self.config["ACTIVE_MODEL_FILENAME"]

        self._load_artifacts()
        self.validator = InferenceDataValidator(self.config)

    def _load_artifacts(self):
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found: {self.preprocessor_path}")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # self.preprocessor = joblib.load(self.preprocessor_path)
        self.model = joblib.load(self.model_path)

    def predict(self, input_data: pd.DataFrame):
        """
        Perform inference on validated input data.
        """
        # 1️⃣ Validate
        input_data = self.validator.validate_dataframe(input_data)

        # 2️⃣ Transform
        # transformed_data = self.preprocessor.transform(input_data)
        # here i need only model which was saved as (preprocessor + model)
        # 3️⃣ Predict    
        preds = self.model.predict(input_data)

        probs = None
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(input_data)[:, 1]

        return preds, probs

if __name__ == "__main__":
    predictor = ModelPredictor()

    sample = pd.DataFrame([{
        "CreditScore": 650,
        "Age": 40,
        "Balance": 75000,
        "EstimatedSalary": 60000,
        "Geography": "France",
        "Gender": "Male",
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "Tenure": 5
    }])

    preds, probs = predictor.predict(sample)

    print("Prediction:", preds)
    print("Probability:", probs)
