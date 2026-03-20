import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from main import load_model, predict_sentiment

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup, clean up on shutdown."""
    logger.info("Loading sentiment model...")
    load_model()
    logger.info("Model loaded successfully.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Sentiment Analysis API",
    description="Predict sentiment (positive / negative / neutral) from text using a fine-tuned BERT model.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — restrict in production via the ALLOWED_ORIGINS env var
import os

allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Schemas ----------

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, examples=["I love this product!"])


class SentimentResponse(BaseModel):
    sentiment: str
    confidence: dict[str, float]


# ---------- Routes ----------

@app.get("/")
async def root():
    return {"message": "Welcome to the Sentiment Analysis API. Visit /docs for interactive documentation."}


@app.get("/health")
async def health():
    """Health check endpoint for Docker / orchestrators."""
    return {"status": "healthy"}


@app.post("/predict", response_model=SentimentResponse)
def predict_api(payload: TextInput):
    """Predict the sentiment of the given text."""
    try:
        result = predict_sentiment(payload.text)
        return result
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
