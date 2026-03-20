import logging

import pandas as pd

logger = logging.getLogger(__name__)


class InferenceDataValidator:
    """Validates incoming inference data against the training schema."""

    def __init__(self, config: dict):
        self.expected_features = (
            config["NUMERICAL_FEATURES"]
            + config["CATEGORICAL_FEATURES"]
            + config["PRE_ENCODED_FEATURES"]
        )

    def validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validates and reorders input DataFrame to match training schema."""
        if df.empty:
            raise ValueError("Inference data is empty.")

        # Extra columns → drop (safe)
        extra_cols = set(df.columns) - set(self.expected_features)
        if extra_cols:
            logger.warning("Dropping extra columns from input: %s", extra_cols)
            df = df.drop(columns=list(extra_cols))

        # Missing columns → hard fail
        missing_cols = set(self.expected_features) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required features: {missing_cols}")

        # Enforce column order (important for sklearn)
        df = df[self.expected_features]

        return df
