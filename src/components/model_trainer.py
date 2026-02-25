import json
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

import wandb
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    )
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.utils.paths import get_config_path, get_model_path
from src.utils.config_loader import load_config
from src.components.metrics import evaluate_model


class ModelTrainer:
    def __init__(self):
        self.config = load_config(get_config_path("paths_config.yaml"))

        base_model_dir = get_model_path()
        base_model_dir.mkdir(parents=True, exist_ok=True)

        self.best_model_path = base_model_dir / self.config.get("MODEL_FILENAME", "best_model.joblib")
        self.preprocessor_path = base_model_dir / self.config.get("PREPROCESSOR_FILENAME", "preprocessor.joblib")
        # self.metrics_path = base_model_dir / self.config.get("METRICS_FILENAME", "metrics.json")
        self.active_model_path = base_model_dir / self.config.get("ACTIVE_MODEL_FILENAME", "active_model.joblib")
        
        metrics_dir  = base_model_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # Timestamped filename
        self.timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.metrics_filename = f"metrics_{self.timestamp}.json"
        self.metrics_path = metrics_dir / self.metrics_filename

        self.random_state = self.config.get("RANDOM_STATE", 42)
        self.selection_metric = self.config.get("MODEL_SELECTION_METRIC", "f1_score")

    # Models
    def get_model_instance(self, model_name, params):
        models_map = {
            "logistic_regression": LogisticRegression,
            "random_forest": RandomForestClassifier,
            "gradient_boosting": GradientBoostingClassifier,
            "xgboost": XGBClassifier,
            "adaboost": AdaBoostClassifier,
            "decision_tree": DecisionTreeClassifier,
        }

        if model_name not in models_map:
            raise ValueError(f"Model {model_name} not supported. Choose from {list(models_map.keys())}")

        model_class = models_map[model_name]
        return model_class(**params)

    # Training
    def initiate_model_trainer(self, X_train, y_train, X_test, y_test, model=None):

        if not self.preprocessor_path.exists():
            raise FileNotFoundError("Preprocessor not found. Run data transformation first.")

        preprocessor = joblib.load(self.preprocessor_path)

        all_metrics = {}
        best_score = -1
        best_pipeline = None
        best_model_name = None

        models_config = self.config["MODELS"]

        for model_name, model_info in models_config.items():
            if not model_info.get("enabled", False):
                continue

            print(f"\n Training {model_name}")

            params = model_info.get("params", {})
            model = self.get_model_instance(model_name, params)

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])
        
        # Weights & Bias
            try:
                wandb.init(
                    project="mlops-classification",
                    name=f"{model_name}-run",
                    reinit = True,
                    config={
                        "model": model_name,
                        **params,
                        "selection_metric": self.selection_metric
                    }
                )
                wandb_enabled = True
            except Exception as e:
                print(f"W&B init failed: {e}")
                wandb_enabled = False

            # Train
            pipeline.fit(X_train, y_train)

            # Evaluate
            y_pred = pipeline.predict(X_test)
            metrics = evaluate_model(y_test, y_pred)

            print(f"{model_name} metrics: {metrics}")

            if wandb_enabled:
                wandb.log(metrics)
            
            all_metrics[model_name] = metrics

            current_score = float(metrics.get(self.selection_metric, 0.0))

            if current_score > best_score:
                best_score = current_score
                best_pipeline = pipeline
                best_model_name = model_name
            
            if wandb_enabled:
                wandb.finish()

        # Save metrics json
        final_metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "selection_metric": self.selection_metric,
            "best_model": best_model_name,
            "best_score": best_score,
            "all_models": all_metrics
        }
        
        with open(self.metrics_path, "w") as fh:
            json.dump(final_metrics, fh, indent=2)
        
        print(f"Metrics saved to: {self.metrics_path}")

        # Save active model 
        joblib.dump(best_pipeline, self.active_model_path)

        # Save best_model
        joblib.dump(best_pipeline, self.best_model_path)

        print(f"\n🏆 BEST MODEL: {best_model_name} ({best_score:.4f})")
        print(f"Saved to: {self.best_model_path}")

        return all_metrics