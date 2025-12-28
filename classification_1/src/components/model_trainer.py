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
        self.metrics_path = base_model_dir / self.config.get("METRICS_FILENAME", "metrics.json")
        self.active_model_path = base_model_dir / self.config.get("ACTIVE_MODEL_FILENAME", "active_model.joblib")
        self.random_state = self.config.get("RANDOM_STATE", 42)

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test, model=None):
        try:
            wandb.init(
                project="mlops-classification",
                name="logreg-train",
                config={
                    "model": "LogisticRegression",
                    "max_iter": 500,
                    "solver": "lbfgs",
                    "class_weight":"balanced",
                    "random_state": self.random_state,
                    "selection_metric": "f1_score"
                }
            )
            wandb_enabled = True
        except Exception as e:
            print(f"W&B init failed: {e}")
            wandb_enabled = False


        if not self.preprocessor_path.exists():
            raise FileNotFoundError("Preprocessor not found. Run data transformation first.")

        preprocessor = joblib.load(self.preprocessor_path)

        if model is None:
            model = LogisticRegression(
                max_iter=300,
                solver="lbfgs",
                class_weight="balanced",
                random_state=self.random_state
            )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        print("Starting training...")
        pipeline.fit(X_train, y_train)
        print("Training finished")

        y_pred = pipeline.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        print(f"metrics: {metrics}")

        if wandb_enabled:
            wandb.log(metrics)

        SELECTION_METRIC = "f1_score"

        if SELECTION_METRIC not in metrics:
            raise KeyError(f"Selection metric '{SELECTION_METRIC}' not found in metrics")

        current_score = float(metrics.get(SELECTION_METRIC, 0.0))

        joblib.dump(pipeline, self.active_model_path)

        if wandb_enabled:
            active_artifact = wandb.Artifact(
                name="active-model",
                type="model"
            )
            active_artifact.add_file(self.active_model_path)
            wandb.log_artifact(active_artifact)

        print(f"Saved active pipeline: {self.active_model_path}")


        if self.metrics_path.exists():
            try:
                prev = json.loads(self.metrics_path.read_text())
                prev_best = float(prev.get("best_score", 0.0))
            except Exception:
                prev_best = 0.0
        else:
            prev_best = 0.0

        if current_score > prev_best:
            joblib.dump(pipeline, self.best_model_path)

            if wandb_enabled:
                best_artifact = wandb.Artifact(
                    name="best-model",
                    type="model",
                    metadata={
                        "selection_metric": SELECTION_METRIC,
                        "best_score": current_score
                    }
                )
                best_artifact.add_file(self.best_model_path)
                wandb.log_artifact(best_artifact)

            tmpfd, tmpname = tempfile.mkstemp()
            os.close(tmpfd)
            with open(tmpname, "w") as fh:
                json.dump({"selection_metric": SELECTION_METRIC,
                           "best_score":current_score,
                           "metrics":metrics,
                           "timestamp": datetime.utcnow().isoformat(),
                           }, fh, indent=2)
            shutil.move(tmpname, self.metrics_path)

            print(f"New BEST model saved: {self.best_model_path} ({prev_best:.4f} → {current_score:.4f})")
        else:
            print(f"No improvement (best={prev_best:.4f}, current={current_score:.4f})")

        if wandb_enabled:
            wandb.finish()
        return metrics
