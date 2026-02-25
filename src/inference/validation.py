# src/inference/validation.py

import pandas as pd


class InferenceDataValidator:
    """Validates incoming inference data using training schema."""

    def __init__(self, config: dict):
        self.expected_features = (
            config["NUMERICAL_FEATURES"]
            + config["CATEGORICAL_FEATURES"]
            + config["PRE_ENCODED_FEATURES"]
        )

    def validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("Inference data is empty")

        # Extra columns → drop (safe)
        extra_cols = set(df.columns) - set(self.expected_features)
        if extra_cols:
            df = df.drop(columns=list(extra_cols))

        # Missing columns → hard fail
        missing_cols = set(self.expected_features) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required features: {missing_cols}")

        # Column order enforcement (important!)
        df = df[self.expected_features]

        return df
