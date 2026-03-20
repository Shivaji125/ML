import logging

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

LABELS = ["negative", "neutral", "positive"]
MODEL_DIR = "./model"

# Module-level references — populated by load_model()
_tokenizer = None
_model = None
_device = None


def load_model(model_dir: str = MODEL_DIR) -> None:
    """Load tokenizer and model from a local directory. Called once at app startup."""
    global _tokenizer, _model, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", _device)

    _tokenizer = AutoTokenizer.from_pretrained(model_dir)
    _model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    _model.to(_device)
    _model.eval()


def predict_sentiment(text: str) -> dict:
    """
    Return the predicted sentiment label and per-class confidence scores.

    Returns
    -------
    dict
        {"sentiment": "positive", "confidence": {"negative": 0.02, "neutral": 0.05, "positive": 0.93}}
    """
    if _model is None or _tokenizer is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    inputs = _tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits

    probabilities = torch.softmax(logits, dim=-1).squeeze().tolist()
    predicted_class = int(torch.argmax(logits, dim=-1).item())

    confidence = {label: round(prob, 4) for label, prob in zip(LABELS, probabilities)}

    return {
        "sentiment": LABELS[predicted_class],
        "confidence": confidence,
    }
