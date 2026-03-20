from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChurnInput(BaseModel):
    """Input schema for churn prediction."""

    CreditScore: int = Field(..., ge=300, le=900)
    Age: int = Field(..., ge=18, le=100)
    Balance: float = Field(..., ge=0)
    EstimatedSalary: float = Field(..., ge=0)

    Geography: Literal["France", "Germany", "Spain"]
    Gender: Literal["Male", "Female"]

    NumOfProducts: int = Field(..., ge=1, le=4)
    HasCrCard: Literal[0, 1]
    IsActiveMember: Literal[0, 1]
    Tenure: int = Field(..., ge=0, le=10)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "CreditScore": 650,
                    "Age": 42,
                    "Balance": 1200.0,
                    "EstimatedSalary": 60000.0,
                    "Geography": "France",
                    "Gender": "Male",
                    "NumOfProducts": 1,
                    "HasCrCard": 1,
                    "IsActiveMember": 1,
                    "Tenure": 5,
                }
            ]
        }
    }


class ChurnPrediction(BaseModel):
    """Output schema for churn prediction."""

    prediction: int
    probability: Optional[float] = None
