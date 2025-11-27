import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
from pathlib import Path
import json

from src.utils.paths import get_config_path, get_model_path
from src.utils.config_loader import load_config
from src.components.metrics import evaluate_model


class ModelTrainer:
    """Trains the final model and evaluates performance."""
    def __init__(self):
        self.config = load_config(get_config_path('paths_config.yaml'))
        c = self.config
        self.trained_model_path = get_model_path() / c['MODEL_FILENAME']

        self.metrics_path = get_model_path() / 'metrics.json'

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        print("---- Starting Model Training ---")
        try:
            # USe a basic model
            model = LogisticRegression(random_state=self.config['RANDOM_STATE'])
            model.fit(X_train, y_train)

            # 1. Prediction
            y_test_pred = model.predict(X_test)

            # 2. Evaluation using the externalized function
            metrics = evaluate_model(y_test, y_test_pred)
            test_accuracy = metrics['accuracy']

            print(f"Model Training complete.")
            print(f"Test Metrics: {metrics}")

            # 3. Save Model Artifact
            self.trained_model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.trained_model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model saved at: {self.trained_model_path}")

            # 4. Save Metrics Artifact
            with open(self.metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            print(f"Metrics saved at: {self.metrics_path}")

            return test_accuracy
        
        except Exception as e:
            print(f"Error during model training: {e}")
            raise e