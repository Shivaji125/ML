"""
Tests for the sentiment analysis API.

To run without a GPU / model download, these tests mock the model.
For integration tests with the real model, set RUN_INTEGRATION_TESTS=1.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add backend to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


# ---------- Unit tests (no model required) ----------


class TestPredictSentiment:
    """Test the predict_sentiment function with a mocked model."""

    @patch("main._model")
    @patch("main._tokenizer")
    @patch("main._device", "cpu")
    def test_returns_dict_with_required_keys(self, mock_tokenizer, mock_model):
        import torch
        from main import predict_sentiment, LABELS

        # Mock tokenizer output
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 2023, 2003, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }

        # Mock model output — logits for [negative, neutral, positive]
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.1, 0.2, 5.0]])
        mock_model.return_value = mock_output

        result = predict_sentiment("I love this!")

        assert "sentiment" in result
        assert "confidence" in result
        assert result["sentiment"] in LABELS
        assert isinstance(result["confidence"], dict)
        assert set(result["confidence"].keys()) == set(LABELS)

    @patch("main._model")
    @patch("main._tokenizer")
    @patch("main._device", "cpu")
    def test_positive_sentiment(self, mock_tokenizer, mock_model):
        import torch
        from main import predict_sentiment

        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 102]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.1, 0.2, 5.0]])  # strong positive
        mock_model.return_value = mock_output

        result = predict_sentiment("Great!")
        assert result["sentiment"] == "positive"

    @patch("main._model")
    @patch("main._tokenizer")
    @patch("main._device", "cpu")
    def test_negative_sentiment(self, mock_tokenizer, mock_model):
        import torch
        from main import predict_sentiment

        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 102]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[5.0, 0.2, 0.1]])  # strong negative
        mock_model.return_value = mock_output

        result = predict_sentiment("Terrible!")
        assert result["sentiment"] == "negative"

    @patch("main._model")
    @patch("main._tokenizer")
    @patch("main._device", "cpu")
    def test_confidence_scores_sum_to_one(self, mock_tokenizer, mock_model):
        import torch
        from main import predict_sentiment

        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 102]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[1.0, 2.0, 3.0]])
        mock_model.return_value = mock_output

        result = predict_sentiment("Some text")
        total = sum(result["confidence"].values())
        assert abs(total - 1.0) < 0.01

    def test_predict_raises_if_model_not_loaded(self):
        """Calling predict before load_model should raise RuntimeError."""
        import main

        # Save originals and force None
        orig_model, orig_tok = main._model, main._tokenizer
        main._model = None
        main._tokenizer = None

        try:
            with pytest.raises(RuntimeError, match="Model not loaded"):
                main.predict_sentiment("hello")
        finally:
            main._model = orig_model
            main._tokenizer = orig_tok


# ---------- API tests (FastAPI TestClient, model mocked) ----------


class TestAPI:
    """Test FastAPI routes."""

    @patch("main.load_model")  # skip real model loading in lifespan
    @patch("main.predict_sentiment")
    def test_predict_endpoint(self, mock_predict, mock_load):
        from fastapi.testclient import TestClient
        from app import app

        mock_predict.return_value = {
            "sentiment": "positive",
            "confidence": {"negative": 0.02, "neutral": 0.05, "positive": 0.93},
        }

        with TestClient(app) as client:
            response = client.post("/predict", json={"text": "I love this!"})

        assert response.status_code == 200
        data = response.json()
        assert data["sentiment"] == "positive"
        assert "confidence" in data

    @patch("main.load_model")
    def test_health_endpoint(self, mock_load):
        from fastapi.testclient import TestClient
        from app import app

        with TestClient(app) as client:
            response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    @patch("main.load_model")
    def test_root_endpoint(self, mock_load):
        from fastapi.testclient import TestClient
        from app import app

        with TestClient(app) as client:
            response = client.get("/")

        assert response.status_code == 200

    @patch("main.load_model")
    def test_predict_empty_text_rejected(self, mock_load):
        from fastapi.testclient import TestClient
        from app import app

        with TestClient(app) as client:
            response = client.post("/predict", json={"text": ""})

        assert response.status_code == 422  # validation error

    @patch("main.load_model")
    def test_predict_missing_text_rejected(self, mock_load):
        from fastapi.testclient import TestClient
        from app import app

        with TestClient(app) as client:
            response = client.post("/predict", json={})

        assert response.status_code == 422
