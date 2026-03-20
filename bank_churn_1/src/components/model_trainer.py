import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.components.metrics import evaluate_model
from src.utils.config_loader import load_config
from src.utils.paths import get_config_path, get_model_path

logger = logging.getLogger(__name__)

# Optional wandb import
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.info("wandb not installed — experiment tracking disabled.")

# Optional xgboost import
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.info("xgboost not installed — XGBoost model disabled.")


class ModelTrainer:
    def __init__(self):
        self.config = load_config(get_config_path("paths_config.yaml"))

        base_model_dir = get_model_path()
        base_model_dir.mkdir(parents=True, exist_ok=True)

        self.best_model_path = base_model_dir / self.config.get(
            "MODEL_FILENAME", "best_model.joblib"
        )
        self.preprocessor_path = base_model_dir / self.config.get(
            "PREPROCESSOR_FILENAME", "preprocessor.joblib"
        )
        self.active_model_path = base_model_dir / self.config.get(
            "ACTIVE_MODEL_FILENAME", "active_model.joblib"
        )

        metrics_dir = base_model_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        self.timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.metrics_filename = f"metrics_{self.timestamp}.json"
        self.metrics_path = metrics_dir / self.metrics_filename

        self.random_state = self.config.get("RANDOM_STATE", 42)
        self.selection_metric = self.config.get("MODEL_SELECTION_METRIC", "f1_score")

    def get_model_instance(self, model_name: str, params: dict):
        """Returns a model instance for the given name and params."""
        models_map = {
            "logistic_regression": LogisticRegression,
            "random_forest": RandomForestClassifier,
            "gradient_boosting": GradientBoostingClassifier,
            "adaboost": AdaBoostClassifier,
            "decision_tree": DecisionTreeClassifier,
        }

        # Add xgboost only if available
        if XGBOOST_AVAILABLE:
            models_map["xgboost"] = XGBClassifier

        if model_name not in models_map:
            raise ValueError(
                f"Model '{model_name}' not supported. Choose from {list(models_map.keys())}"
            )

        return models_map[model_name](**params)

    def initiate_model_trainer(
        self, X_train, y_train, X_test, y_test, model=None
    ) -> dict:
        """Trains all enabled models, evaluates, selects the best, and saves artifacts."""

        if not self.preprocessor_path.exists():
            raise FileNotFoundError(
                "Preprocessor not found. Run data transformation first."
            )

        preprocessor = joblib.load(self.preprocessor_path)

        all_metrics: dict = {}
        best_score = -1.0
        best_pipeline = None
        best_model_name = None

        models_config = self.config["MODELS"]

        for model_name, model_info in models_config.items():
            if not model_info.get("enabled", False):
                logger.info("Skipping disabled model: %s", model_name)
                continue

            # Skip xgboost if not installed
            if model_name == "xgboost" and not XGBOOST_AVAILABLE:
                logger.warning("Skipping xgboost — not installed.")
                continue

            logger.info("Training %s...", model_name)

            params = model_info.get("params", {})
            model_instance = self.get_model_instance(model_name, params)

            pipeline = Pipeline(
                [("preprocessor", preprocessor), ("model", model_instance)]
            )

            # W&B init (optional)
            wandb_enabled = False
            if WANDB_AVAILABLE:
                try:
                    wandb.init(
                        project="mlops-classification",
                        name=f"{model_name}-run",
                        reinit=True,
                        config={
                            "model": model_name,
                            **params,
                            "selection_metric": self.selection_metric,
                        },
                    )
                    wandb_enabled = True
                except Exception as e:
                    logger.warning("W&B init failed: %s", e)

            # Train
            pipeline.fit(X_train, y_train)

            # Evaluate on test set
            y_pred = pipeline.predict(X_test)

            # Get probability scores for ROC-AUC
            y_prob = None
            if hasattr(pipeline, "predict_proba"):
                y_prob = pipeline.predict_proba(X_test)[:, 1]

            # Get train predictions for overfitting detection
            y_train_pred = pipeline.predict(X_train)

            metrics = evaluate_model(
                y_test,
                y_pred,
                y_prob=y_prob,
                y_train_true=y_train,
                y_train_pred=y_train_pred,
            )

            logger.info("%s metrics: %s", model_name, {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in metrics.items()
                if k != "confusion_matrix"
            })

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

        if best_pipeline is None:
            raise RuntimeError("No models were trained. Check MODELS config.")

        # Save metrics
        final_metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "selection_metric": self.selection_metric,
            "best_model": best_model_name,
            "best_score": best_score,
            "all_models": all_metrics,
        }

        with open(self.metrics_path, "w") as fh:
            json.dump(final_metrics, fh, indent=2)
        logger.info("Metrics saved to: %s", self.metrics_path)

        # Save active model and best model
        joblib.dump(best_pipeline, self.active_model_path)
        joblib.dump(best_pipeline, self.best_model_path)

        logger.info(
            "BEST MODEL: %s (%s=%.4f) — saved to %s",
            best_model_name, self.selection_metric, best_score, self.best_model_path,
        )

        return all_metrics
