from fastapi import FastAPI, HTTPException
import pandas as pd

from src.inference.predictor import ModelPredictor
from src.inference.schemas import ChurnInput, ChurnPrediction

app = FastAPI(
    title="Churn Predictor API",
    version= "1.0.0"
)

predictor = ModelPredictor()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=ChurnPrediction)
def predict(input_data: ChurnInput):
    try:
        df  = pd.DataFrame([input_data.model_dump()])
        preds, probs = predictor.predict(df)

        return ChurnPrediction(
            prediction = int(preds[0]),
            probability = float(probs[0]) if probs is not None else None
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))