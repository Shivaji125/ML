import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.inference.predictor import ModelPredictor

@patch("src.inference.predictor.joblib.load")
@patch("src.inference.predictor.Path.exists")
def test_predict_success(mock_exists, mock_joblib_load):
    mock_exists.return_value = True

    mock_model = MagicMock()
    mock_model.pedict.return_value = [1]
    mock_model.predict_proba.return_value = [[0.2, 0.8]]

    mock_joblib_load.return_value = mock_model

    predictor = ModelPredictor()

    df = pd.DataFrame([{
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

    preds, probs = predictor.predict(df)

    assert preds == [1]
    assert probs[0] == 0.8