import logging

import joblib
import pandas as pd

from src.inference.validation import InferenceDataValidator
from src.utils.config_loader import load_config
from src.utils.paths import get_config_path, get_model_path

logger = logging.getLogger(__name__)


class ModelPredictor:
    """Loads trained artifacts and performs inference."""

    def __init__(self):
        self.config = load_config(get_config_path("paths_config.yaml"))

        # FIX: use get_model_path() for absolute path (consistent with rest of codebase)
        model_dir = get_model_path()
        self.preprocessor_path = model_dir / self.config["PREPROCESSOR_FILENAME"]
        self.model_path = model_dir / self.config["ACTIVE_MODEL_FILENAME"]

        self._load_artifacts()
        self.validator = InferenceDataValidator(self.config)

    def _load_artifacts(self) -> None:
        """Load the saved pipeline (preprocessor + model bundled)."""
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found: {self.preprocessor_path}")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # The saved pipeline already includes (preprocessor + model)
        self.model = joblib.load(self.model_path)
        logger.info("Model loaded from %s", self.model_path)

    def predict(self, input_data: pd.DataFrame) -> tuple:
        """Perform inference on validated input data.

        Returns:
            tuple: (predictions array, probability array or None)
        """
        # 1. Validate input
        input_data = self.validator.validate_dataframe(input_data)

        # 2. Predict (pipeline handles preprocessing internally)
        preds = self.model.predict(input_data)

        probs = None
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(input_data)[:, 1]

        return preds, probs
