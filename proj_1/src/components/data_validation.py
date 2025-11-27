import pandas as pd
import numpy as np
import json
from pathlib import Path

# Import our robust utilities
from src.utils.paths import get_data_path, get_config_path
from src.utils.config_loader import load_config


# -- 1. Define the Expected Schema ---
class DataSchema:
    """Defines the expected data schema and validation rules."""

    # # Expected columns and data types
    # EXPECTED_DTYPES = {
    #     'feature_A': (np.int64, np.float64),
    #     'feature_B': (object,),
    #     'target': (np.int64,)
    # }

    # 1. Define the Groups of Features
    # List all features that should be numeric (int or float)
    NUMERIC_FEATURES = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary',]
    # Features that are already encode/discrete
    PRE_ENCODED_NUMERIC_FEATURES = ['NumOfProducts','HasCrCard','IsActiveMember','Tenure',]
    # List all features that should be treated as categorical objects (strings/objects)
    CATEGORICAL_OHE_FEATURES = ['Geography','Gender',]
    # Define the Target Variable (always handled separately)
    TARGET_COLUMN = 'Exited'

    # 2. Define the Expected Data Types for the Groups
    # We maintain the actual expected data types here for the validation logic
    NUMERIC_DTYPES = (np.int64, np.float64)
    CATEGORICAL_DTYPES = (object,)
    TARGET_DTYPES = (np.int64,)

    # 3. Construct the final Expected dictionary dynamically
    # This makes the validation loop easier to write and maintain
    EXPECTED_DTYPES = {}

    # Combine all truly numeric features for validation check
    ALL_NUMERIC_FEATURES = NUMERIC_FEATURES + PRE_ENCODED_NUMERIC_FEATURES
    # Add numeric features
    for col in ALL_NUMERIC_FEATURES:
        EXPECTED_DTYPES[col] = NUMERIC_DTYPES
    # Add categorical OHE features
    for col in CATEGORICAL_OHE_FEATURES:
        EXPECTED_DTYPES[col] = CATEGORICAL_DTYPES
    # Add the target column
    EXPECTED_DTYPES[TARGET_COLUMN] = TARGET_DTYPES

    # Construct dictionary: { column: (min_value, max_value)}
    NUMERIC_CONSTRAINTS = {
        'feature_A': (1,10)  # Example: feature_A must be between 1 and 10
    }

    # Allowed unique categories values
    CATEGORICAL_CONSTRAINTS = {
        'feature_B': ['X', 'Y', 'Z']  # Example: only these values are allowed
    }


# --- 2. Data Validation Logic ---
class DataValidation:
    def __init__(self, schema=DataSchema()):
        #  Load configuration (needed to find the input file)
        self.config = load_config(get_config_path('paths_config.yaml'))

        # Construct the input data path (validation data should be the split training data) 
        c =self.config
        self.input_data_path:  Path = get_data_path(c['PROCESSED_DIR']) / c['TRAIN_DATA_FILENAME']

        self.schema = schema
        self.validation_report = {}
        self.is_valid = True

    def _check_schema_compliance(self, df: pd.DataFrame):
        """Checks for missing columns and incorrect data types."""
        print(" -> Checking Schema Compliance...")

        expected_cols = set(self.schema.EXPECTED_DTYPES.keys())
        current_cols = set(df.columns)

        # 0. Extra Column Removal and Report
        extra_cols = list(current_cols - expected_cols)
        if extra_cols:
            print(f"**Action:** Removed extra columns: {extra_cols}")
            self.validation_report['Removed_Extra_Columns'] = extra_cols
            # Remove the extra columns from the DataFrame
            df = df.drop(columns=[extra_cols], axis=1, errors='ignore')
            current_cols = set(df.columns)   # !update current columns after dropping
    
        # 1. Column Presence Check
        missing_cols = list(expected_cols - current_cols)
        if missing_cols:
            self.validation_report['Misisng_Columns'] = missing_cols
            self.is_valid = False

        # 2. Data Type Check
        incorrect_types = {}
        for col, dtypes in self.schema.EXPECTED_DTYPES.items():
            if col in df.columns and df[col].dtype not in dtypes:
                incorrect_types[col] = f"Expected one of {dtypes}, got {df[col].dtype}"
                self.is_valid = False

        if incorrect_types:
            self.validation_report['Incorrect_Data_Types'] = incorrect_types

    
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
        """Runs all vaidation checks on the pre_defined training data file."""
        print(f"---- Starting Data Validation on: {self.input_data_path.name}")

        try:
            if not self.input_data_path.exists():
                raise FileNotFoundError(f"Input data not found at {self.input_data_path}")
            
            df = pd.read_csv(self.input_data_path)

            # Reset validation ststus before starting
            self.validation_report = {}
            self.is_valid = True

            # Run checks
            self._check_schema_compliance(df)
            # self._check_data_constraints(df)

            # Final Report
            if self.is_valid:
                print("---- Data Validation PASSES! ---")
            else:
                print("!!! Data Validation FAILED! Quality issues detected. !!!")
                print("Validation Report:")
                print(json.dumps(self.validation_report, indent=4))

            return self.is_valid
        
        except Exception as e:
            self.validation_report["Execution_Error"] = str(e)
            print(f"An error occured during validation: {e}")
            raise e