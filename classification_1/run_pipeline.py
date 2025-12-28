import sys
import os
from pathlib import Path
import pandas as pd

from src.components.data_ingestion import  DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils.paths import get_data_path

from src.utils.paths import get_data_path, get_config_path
from src.utils.config_loader import load_config

config = load_config(get_config_path("paths_config.yaml"))

def run_pipeline():
    """Executes the end to end modular ML pipeline."""
    print("\n --Starting End to End Ml Pipeline Execution.")

    
    ingestor = DataIngestion()
    train_path, test_path = ingestor.initiate_data_ingestion()

    print("Data Validation Starting")
    validator = DataValidation(config)
    is_data_valid = validator.initiate_data_validation()

    if not is_data_valid:
        print("Pipeline halted due to criticall Data Quality Issues.")
        sys.exit(1) # Stop Execution if data quality fails
    print("Data Validation completed successfully")

    transfomer = DataTransformation()
    X_train, y_train, X_test, y_test, preprocessor_path = \
        transfomer.initiate_data_transformation(train_path, test_path)

    trainer = ModelTrainer()
    metrics = trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)

    print("Pipeline completed successfully.")
    return metrics
    
    

if __name__ == "__main__":
    try: 
        run_pipeline()
    except Exception as e:
        print("Pipeline FAILED")
        sys.exit(1)