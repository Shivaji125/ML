import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

# --- Define Paths (Must match your project structure/config) ---
# Assuming 'models' directory is at the project root
PROJECT_ROOT = Path(__file__).parent
print("project root:", PROJECT_ROOT)
MODEL_DIR = PROJECT_ROOT / "models"
PREPROCESSOR_FILENAME = "preprocessor.pkl"
MODEL_FILENAME = "logistic_regression.pkl"

# Define a placeholder for expeted features. In a real system,
# you might load this from a JSON artifact saved during validation/training.
# For simplicity, we define them here based on your schema.

NUMERIC_FEATURES = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary',]
PRE_ENCODED_NUMERIC_FEATURES = ['NumOfProducts','HasCrCard','IsActiveMember','Tenure',]
CATEGORICAL_OHE_FEATURES = ['Geography','Gender',]

# 2. Define the Expected Data Types for the Groups
NUMERIC_DTYPES = (np.int64, np.float64)
CATEGORICAL_DTYPES = (object,)

EXPECTED_FEATURES = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary','Geography','Gender','NumOfProducts','HasCrCard','IsActiveMember','Tenure',]
# you should include 'feature_C' are any other features used in transformation
# like feature engineering.

# --- Data Type Mapping (For enforcement) ---
# Define expected dtypes based on your DataSchema (excluding 'target')
EXPECTED_DTYPES = {}

# Combine all truly numeric features for validation check
ALL_NUMERIC_FEATURES = NUMERIC_FEATURES + PRE_ENCODED_NUMERIC_FEATURES
# Add numeric features
for col in ALL_NUMERIC_FEATURES:
    EXPECTED_DTYPES[col] = NUMERIC_DTYPES
# Add categorical OHE features
for col in CATEGORICAL_OHE_FEATURES:
    EXPECTED_DTYPES[col] = CATEGORICAL_DTYPES


class ModelPredictor:
    """Handles loading artifacts and performing schema-enforced predictions."""
    def __init__(self):
        
        # 1. Load Artifacts
        try:
            preprocessor_full_path = MODEL_DIR / PREPROCESSOR_FILENAME
            model_full_path = MODEL_DIR / MODEL_DIR / MODEL_FILENAME
            
            with open(preprocessor_full_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            with open(model_full_path, 'rb') as f:
                self.model = pickle.load(f)
            
            print("Artifacts loaded successfully.")

        except Exception as e:
            print(f"CRITICAL ERROR: Artifacts not found. Run training pipeline first: {e}")
            raise e
        
        self.expected_features = EXPECTED_FEATURES
        self.expected_dtypes = EXPECTED_DTYPES

    def _schema_enforcement(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs structural checks (missing/extra columns, data types)
        and prepares the DataFrame for transformation.
        """
        current_cols = set(df.columns)

        # 1. Extra Columns (CRITICAL)
        extra_cols = list(current_cols - set(self.expected_features))
        if extra_cols:
            print(f"[Schema Action] Dropping extra columns: {extra_cols}")
            df = df.drop(columns=extra_cols, errors='ignore')
        
        # 2. Missing Columns (CRITICAL)
        missing_cols = list(set(self.expected_features) - set(df.columns))
        if missing_cols:
            raise ValueError(f"Schema Mismatch: Missing required features: {missing_cols}")
        
        # Ensure the column order is correct before transformation
        df = df[self.expected_features]

        # 3. Data Types (HIGH)
        for col, expected_type in self.expected_dtypes.items():
            # Check if the column is present (it should be at this point)
            if col in df.columns and not pd.api.types.is_dtype_equal(df[col].dtype, expected_type):
                try:
                    # Attempt to force correct data type (e.g., string to float)
                    if expected_type == object:
                        df[col] = df[col].astype(str)
                    elif expected_type == np.number:
                        df[col] = pd.to_numeric(df[col], errors='raise')
                except Exception as e:
                    # Fail prediction if data cannot be coerced
                    raise TypeError(f"Data Type Error in column '{col}: Could not convert to required type ({expected_type}). Error: {e}")
                
        return df
    
    def predict(self, new_data: pd.DataFrame) -> np.ndarray:
        """Applies schema enforcement, transformation, and returns predictions."""
        print(f"\n--- Starting Inference on {len(new_data)} row(s) ---")

        try:
            # 1. Schema Enforcement and Cleaning
            data_to_predict = self._schema_enforcement(new_data.copy()) # Use a copy to prevent side effects

            # 2. Transformation (using the fitted preprocessor)
            data_transformed = self.preprocessor.transform(data_to_predict)

            # 3. Prediction
            predictions = self.model.predict(data_transformed)

            return predictions
        
        except [ValueError, TypeError, KeyError] as e:
            print(f"Inference failed dur to structural data error: {e}")
            # In a deployed service, you might return a default value or an error code.
            return np.array([f"Prediction Failed: {e}"])
        except Exception as e:
            print(f"An unexpected error occured during inference: {e}")
            return np.array([f'Prediction Failed: {e}'])
        
if __name__ == "__main__":
    # --- 1. Initialize the Predictor ---
    predictor = ModelPredictor()

    # --- 2. Simulate Real-World Input Data ---
    # Case A: A perfect customer (Expected to be 0: No Churn)
    data_point_1 = {
        'CreditScore': 650, 'Age': 40, 'Balance': 75000.0, 'EstimatedSalary': 60000.0,
        'NumOfProducts': 1, 'HasCrCard': 1, 'IsActiveMember': 1, 'Tenure': 5,
        'Geography': 'France', 'Gender': 'Male',
    }

    # Case B: A high-risk customer (Low Credit, High Balance, expected 1: Churn)
    data_point_2 = {
        'CreditScore': 400, 'Age': 55, 'Balance': 150000.0, 'EstimatedSalary': 120000.0,
        'NumOfProducts': 3, 'HasCrCard': 0, 'IsActiveMember': 0, 'Tenure': 1,
        'Geography': 'Germany', 'Gender': 'Female',
    }
    
    # Case C: Data with an extra column and a dtype error (Test Schema Enforcement)
    data_point_3_error = {
        'CreditScore': 700, 'Age': '45', 'Balance': 0.0, 'EstimatedSalary': 50000.0,
        'NumOfProducts': 2, 'HasCrCard': 1, 'IsActiveMember': 1, 'Tenure': 8,
        'Geography': 'Spain', 'Gender': 'Male',
        'Customer_ID': 'XYZ123' # Extra column
    }
    
    # Combine into a single DataFrame for the prediction function
    inference_df = pd.DataFrame([data_point_1, data_point_2, data_point_3_error])

    # --- 3. Get Predictions ---
    final_predictions = predictor.predict(inference_df)

    # --- 4. Display Results ---
    print("\n--- Inference Results Summary ---")
    
    # Reset index for clean display
    inference_df = inference_df.reset_index(drop=True) 
    
    # Create results table
    results_df = pd.DataFrame({
        'Input_Data': [dict(row) for index, row in inference_df.iterrows()],
        'Prediction': final_predictions
    })
    
    print(results_df[['Input_Data', 'Prediction']].to_markdown(index=False, numalign="left"))
    
    # Expected Output Interpretation: 0=No Churn, 1=Churn