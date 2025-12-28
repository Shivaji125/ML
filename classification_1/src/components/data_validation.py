import pandas as pd
import numpy as np
import json
from pathlib import Path

from src.utils.paths import get_data_path, get_config_path
from src.utils.config_loader import load_config

config = load_config(get_config_path("paths_config.yaml"))

# Define the Expected Schema
class DataSchema:
    def __init__(self, config: dict):
        self.NUMERIC_FEATURES = config.get("NUMERIC_FEATURES", [])
        self.PRE_ENCODED_FEATURES = config.get("PRE_ENCODED_FEATURES", [])
        self.CATEGORICAL_FEATURES = config.get("CATEGORICAL_FEATURES", [])
        self.TARGET_COLUMN = config["TARGET_COLUMN"]

        self.NUMERIC_DTYPES = (np.int64, np.float64)
        self.CATEGORICAL_DTYPES = (object,)
        self.TARGET_DTYPES = (np.int64,)

        self.ALL_NUMERIC_FEATURES = self.NUMERIC_FEATURES + self.PRE_ENCODED_FEATURES
        self.EXPECTED_DTYPES = {}

        for col in self.ALL_NUMERIC_FEATURES:
            self.EXPECTED_DTYPES[col] = self.NUMERIC_DTYPES
        for col in self.CATEGORICAL_FEATURES:
            self.EXPECTED_DTYPES[col] = self.CATEGORICAL_DTYPES

        self.EXPECTED_DTYPES[self.TARGET_COLUMN] = self.TARGET_DTYPES

        # Construct dictionary: { column: (min_value, max_value)}
        NUMERIC_CONSTRAINTS = {
            'feature_A': (1,10)  # Example: feature_A must be between 1 and 10
        }

        # Allowed unique categories values
        CATEGORICAL_CONSTRAINTS = {
            'feature_B': ['X', 'Y', 'Z']  # Example: only these values are allowed
        }

# Data Validation logic
class DataValidation:
    def __init__(self, config: dict):
        self.input_data_path = get_data_path(config["PROCESSED_DIR"]) / config["TRAIN_DATA_FILENAME"]
        
        self.schema = DataSchema(config)
        self.validation_report = {}
        self.is_valid = True

    def _check_schema_compliance(self, df: pd.DataFrame):
        """Check for missing columns and incorrect data types."""
        print("-> Checking Schema Compliance.")
        
        expected_cols = set(self.schema.EXPECTED_DTYPES.keys())
        current_cols = set(df.columns)

        # Extra Column Removala and Report
        extra_cols = list(current_cols - expected_cols)
        if extra_cols:
            print(f"Action: Removed extra columns: {extra_cols}")
            self.validation_report["Removed_Extra_Columns"] = extra_cols
            df = df.drop(columns=extra_cols, axis=1, errors='ignore')
            current_cols = set(df.columns)

        # Column presence check
        missing_cols = list(expected_cols - current_cols)
        if missing_cols:
            self.validation_report["Missing_Columns"] = missing_cols
            self.is_valid = False
        
        # Data Type Check
        incorrect_types = {}
        for col, expected_dtypes in self.schema.EXPECTED_DTYPES.items():
            if col in df.columns and df[col].dtype not in expected_dtypes:
                incorrect_types[col] = f"Expected one of {expected_dtypes}, got {df[col].dtype}"
                self.is_valid = False
        
        if incorrect_types:
            self.validation_report["Incorrect_Data_Types"] = incorrect_types

    def _check_data_constraints(self, df: pd.DataFrame):
        """Checks numerical ranges and allowed categorical values."""
        print("     -> Checking Data Constraints (Ranges, Categories)...")

        # 1. Numerical Range Check
        range_violations = {}
        for col, (min_val, max_val) in self.schema.NUMERIC_CONSTRAINTS.items():
            if col in df.columns:
                violated_count  = df[~df[col].between(min_val, max_val)].shape[0]
                if violated_count > 0:
                    range_violations[col] = f"{violated_count} records are outside the range ({min_val}, {max_val})"
                    self.is_valid = False
        if range_violations:
            self.validation_report['Range_Violations'] = range_violations

        # 2. categorical Value Check
        category_violations = {}
        for col, allowed_values in self.schema.CATEGORICAL_CONSTRAINTS.items():
            if col in df.columns:
                unseen_values = set(df[col].unique()) - set(allowed_values)
                if unseen_values:
                    category_violations[col] = f"Found unseen categories: {list(unseen_values)}"
                    self.is_valid = False

        if category_violations:
            self.validation_report['Category_Violations'] = category_violations

    def initiate_data_validation(self) -> bool:
        """Runs all validation check on the pre-defined training data file."""
        print(f"--- Starting Data Validation on: {self.input_data_path.name}")

        try:
            if not self.input_data_path.exists():
                raise FileNotFoundError(f"Input data not found at {self.input_data_path}")
            
            df = pd.read_csv(self.input_data_path)

            self.validation_report = {}
            self.is_valid = True

            # Run checks
            self._check_schema_compliance(df)
            # self._check_data_constraints(df)

            # Final Report
            if self.is_valid:
                print("--- Data Validation Passes! ---")
            else:
                print('!!! Data Validation FAILED! Quality issues detected !!!.')
                print(f"Validation Report: ")
                print(json.dumps(self.validation_report, indent = 4))
        

            return self.is_valid


        except Exception as e:
            self.validation_report["Execution_Error"] = str(e)
            print(f"An error occured during validation: {e}")
            raise e