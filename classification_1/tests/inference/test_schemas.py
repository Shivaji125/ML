import pytest
from pydantic import ValidationError

from src.inference.schemas import ChurnInput

def test_schema_valid_input():
    ChurnInput(
        CreditScore=650,
        Age=40,
        Balance=75000,
        EstimatedSalary=60000,
        Geography="France",
        Gender="Male",
        NumOfProducts=1,
        HasCrCard=1,
        IsActiveMember=1,
        Tenure=5
    )

def test_schema_invalid_input():
    with pytest.raises(ValidationError):
        ChurnInput(
            CreditScore="bad",  # invalid
            Age=40,
            Balance=75000,
            EstimatedSalary=60000,
            Geography="France",
            Gender="Male",
            NumOfProducts=1,
            HasCrCard=1,
            IsActiveMember=1,
            Tenure=5
        )
