import pandas as pd
import pytest

from src.inference.validation import InferenceDataValidator

@pytest.fixture
def config():
    return{
        "NUMERICAL_FEATURES": [
            "CreditScore", "Age", "Balance", "EstimatedSalary", "Tenure"
        ],
        "CATEGORICAL_FEATURES": [
            "Geography", "gender"
        ],
        "PRE_ENCODED_FEATURES": [
            "NumOfProducts", "HasCrCard", "IsActiveMember"
        ],
    }

def test_validation_success(config):
    validator = InferenceDataValidator(config)

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

def test_missing_columns_fail(config):
    validator = InferenceDataValidator(config)

    df = pd.DataFrame([{
        "CreditScore": 650
    }])

    with pytest.raises(ValueError):
        validator.validate_dataframe(df)

def test_extra_columns_are_dropped(config):
    validator = InferenceDataValidator(config)

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
        "Tenure": 5,
        "RandomColumn": 123
    }])

    out = validator.validate_dataframe(df)

    assert "RandomColumn" not in out.columns