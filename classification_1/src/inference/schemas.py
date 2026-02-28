from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class ChurnInput(BaseModel):
    CreditScore: int = Field(...,ge=300, le=900, example=650)
    Age: int = Field(...,ge=18, le=100, example=42)
    Balance: float = Field(..., ge=0, example=1200.0)
    EstimatedSalary: float = Field(..., ge=0, example=60000)

    Geography: Literal["France", "Germany", "Spain"] = Field(..., example="France")
    Gender: Literal["Male", "Female"] = Field(..., example="Male")

    NumOfProducts: int = Field(...,ge=1, le=4, example=1)
    HasCrCard: Literal[0, 1] = Field(..., example=1)
    IsActiveMember: Literal[0, 1] = Field(..., example=1)
    Tenure: int = Field(...,ge =0, le=10, example=5)

class ChurnPrediction(BaseModel):
    prediction: int
    probability: Optional[float]