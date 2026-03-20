import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.config_loader import load_config
from src.utils.paths import get_config_path, get_model_path

logger = logging.getLogger(__name__)


class DataTransformation:
    """Handles feature engineering, scaling, and encoding."""

    def __init__(self):
        self.config = load_config(get_config_path("paths_config.yaml"))
        self.preprocessor_path = get_model_path() / self.config["PREPROCESSOR_FILENAME"]
        self.target_column = self.config["TARGET_COLUMN"]

    def get_data_transformer_object(self) -> ColumnTransformer:
        """Creates the preprocessing ColumnTransformer."""
        numerical_features = self.config["NUMERICAL_FEATURES"]
        categorical_features = self.config["CATEGORICAL_FEATURES"]
        pre_encoded_features = self.config["PRE_ENCODED_FEATURES"]

        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        pre_encoded_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("num_pipeline", num_pipeline, numerical_features),
                ("cat_pipeline", cat_pipeline, categorical_features),
                ("pre_encoded_pipeline", pre_encoded_pipeline, pre_encoded_features),
            ]
        )

        return preprocessor

    def initiate_data_transformation(
        self, train_path: str, test_path: str
    ) -> tuple:
        """Loads data, fits the preprocessor, and saves it.

        Note: The preprocessor is only *fit* here — not used to transform.
        Transformation happens inside the sklearn Pipeline during model training,
        since the pipeline bundles (preprocessor + model) together.
        """
        logger.info("Starting data transformation...")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        X_train = train_df.drop(columns=[self.target_column])
        y_train = train_df[self.target_column]

        X_test = test_df.drop(columns=[self.target_column])
        y_test = test_df[self.target_column]

        preprocessor = self.get_data_transformer_object()

        # Fit the preprocessor on training data (pipeline will call transform internally)
        preprocessor.fit(X_train)

        # Save the fitted preprocessor
        self.preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, self.preprocessor_path)
        logger.info("Preprocessor saved at: %s", self.preprocessor_path)

        return (
            X_train,
            y_train.values,
            X_test,
            y_test.values,
            str(self.preprocessor_path),
        )
