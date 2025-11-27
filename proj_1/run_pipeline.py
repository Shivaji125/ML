import sys
import os
from pathlib import Path
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils.paths import get_data_path


# # --- Setup Dummy Data for First Run ---
# def setup_initial_data():
#     """Creates a dummy raw data file so the pipeline can run immediately."""
#     dummy_data  = {
#         'feature_A': [1,2,3,4,5,6,7,8,9,10],
#         'feature_B': ['X', 'Y', 'X', 'Z', 'Y', 'X', 'Y', 'Z', 'X', 'Y'],
#         'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
#         'extra_col_1': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], # Extra column
#     }
#     dummy_df = pd.DataFrame(dummy_data)

#     raw_data_path = get_data_path('raw') / 'data.csv'
#     raw_data_path.parent.mkdir(parents=True, exist_ok=True)
#     dummy_df.to_csv(raw_data_path, index=False)
#     print(f"Created dummy raw data at: {raw_data_path}")

def run_training_pipeline():
    """Executes the end-to-end modular Ml pipeline."""
    # setup_initial_data()
    print("\n--- Starting End-to-End ML Pipeline Execution ---")

    try:
        # 1. Data Ingestion
        ingestor = DataIngestion()
        train_path, test_path = ingestor.initiate_data_ingestion()

        # 2. Data Validation (Operates on the ingested data)
        validator = DataValidation()
        is_data_valid = validator.initiate_data_validation()

        if not is_data_valid:
            print("Pipeline Halted due to critical Data Quality Issues.")
            sys.exit(1)   # Stop Execution if data quality fails

        # 3. Data Transformation
        transformer = DataTransformation()
        X_train, y_train, X_test, y_test, preprocessor_path = \
        transformer.initiate_data_transformation(train_path, test_path)

        # 4. Model Training
        trainer = ModelTrainer()
        accuracy = trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)

        # print("\n--- Pipeline SUCCESS Summary ---")
        # print(f"Final Test Accuracy: {accuracy:.4f}")
        # print(f"Model Artifacts saved in: {Path(preprocessor_path).parent}")
    
    except Exception as e:
        print(f"\n--- Pipeline FAILED ----")
        print(f"An unexpected error occured: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_training_pipeline()