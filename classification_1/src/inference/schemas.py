from typing import List, Optional
from pydantic import BaseModel, Field


class ChurnInput(BaseModel):
    CreditScore: int = Field(..., example=650)
    Age: int = Field(..., example=42)
    Balance: float = Field(..., example=1200.0)
    EstimatedSalary: float = Field(..., example=60000)

    Geography: str = Field(..., example="France")
    Gender: str = Field(..., example="Male")

    NumOfProducts: int = Field(..., example=1)
    HasCrCard: int = Field(..., example=1)
    IsActiveMember: int = Field(..., example=1)
    Tenure: int = Field(..., example=5)

class ChurnPrediction(BaseModel):
    prediction: int
    probability: Optional[float]