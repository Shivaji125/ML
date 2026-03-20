# Bank Churn Classification — MLOps Pipeline

End-to-end ML pipeline for predicting bank customer churn, with modular components, config-driven training, experiment tracking (W&B), and a FastAPI serving layer.

## Project Structure

```
classification_1/
├── config/
│   └── paths_config.yaml      # All paths, features, model hyperparameters
├── src/
│   ├── components/
│   │   ├── data_ingestion.py   # Load raw data, train/test split
│   │   ├── data_validation.py  # Schema & constraint checks
│   │   ├── data_transformation.py  # Preprocessing pipeline
│   │   ├── model_trainer.py    # Train, evaluate, select best model
│   │   └── metrics.py          # Classification metrics
│   ├── inference/
│   │   ├── predictor.py        # Load model & predict
│   │   ├── validation.py       # Validate inference inputs
│   │   └── schemas.py          # Pydantic request/response models
│   ├── api/
│   │   └── main.py             # FastAPI app
│   ├── utils/
│   │   ├── paths.py            # Project path helpers
│   │   └── config_loader.py    # YAML config loader
│   └── run_pipeline.py         # End-to-end pipeline runner
├── Dockerfile
├── locustfile.py               # Load testing
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Place your raw data at `data/raw/churn_bank.csv`.

## Run the Training Pipeline

```bash
python -m src.run_pipeline
```

This runs: **Ingestion → Validation → Transformation → Training** and saves the best model to `models/`.

## Start the API

```bash
uvicorn src.api.main:app --reload
```

### Endpoints

- `GET /health` — Health check
- `POST /predict` — Predict churn

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 650,
    "Age": 40,
    "Balance": 75000,
    "EstimatedSalary": 60000,
    "Geography": "France",
    "Gender": "Male",
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "Tenure": 5
  }'
```

## Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## Load Testing

```bash
locust -f locustfile.py --host=http://localhost:8000
```

## Configuration

All settings are in `config/paths_config.yaml` — file paths, feature lists, model hyperparameters, and selection metrics. Enable/disable models by toggling the `enabled` flag.
