from fastapi import FastAPI, HTTPException
import pandas as pd
import asyncio
from contextlib import asynccontextmanager

from src.inference.predictor import ModelPredictor
from src.inference.schemas import ChurnInput, ChurnPrediction

app = FastAPI(
    title="Churn Predictor API",
    version= "1.0.0"
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")
    app.state.predictor = ModelPredictor()
    print("Model loaded successfully")
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)


@app.post("/predict")
async def predict(input_data: ChurnInput):

    try:
        df = pd.DataFrame([input_data.dict()])
        preds, probs = await asyncio.to_thread(
            app.state.predictor.predict,
            df
        )

        return {
            "prediction": preds.tolist(),
            "probability": probs.tolist() if probs is not None else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
