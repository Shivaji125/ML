import asyncio
import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.inference.predictor import ModelPredictor
from src.inference.schemas import ChurnInput, ChurnPrediction

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    logger.info("Loading model...")
    app.state.predictor = ModelPredictor()
    logger.info("Model loaded successfully.")
    yield
    logger.info("Shutting down...")


# FIX: single app creation with lifespan, title, and version all together
app = FastAPI(
    title="Churn Predictor API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": hasattr(app.state, "predictor"),
    }


@app.post("/predict", response_model=ChurnPrediction)
async def predict(input_data: ChurnInput):
    """Predict customer churn probability."""
    try:
        df = pd.DataFrame([input_data.model_dump()])
        preds, probs = await asyncio.to_thread(
            app.state.predictor.predict, df
        )

        return ChurnPrediction(
            prediction=int(preds[0]),
            probability=float(probs[0]) if probs is not None else None,
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal prediction error.")
