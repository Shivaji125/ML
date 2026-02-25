import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import os

# import our new utilities
from src.utils.paths import get_data_path, get_config_path, get_root
from src.utils.config_loader import load_config

class DataIngestion:
    def __init__(self):
        # load config 
        self.config = load_config(get_config_path("paths_config.yaml"))
        self.c = self.config

        self.raw_path = get_data_path(self.c["RAW_DIR"]) / self.c["RAW_DATA_FILENAME"]
        self.train_path = get_data_path(self.c["PROCESSED_DIR"]) / self.c["TRAIN_DATA_FILENAME"]
        self.test_path = get_data_path(self.c["PROCESSED_DIR"]) / self.c["TEST_DATA_FILENAME"]

    def initiate_data_ingestion(self):
        print(f"Loading data from: {self.raw_path}")
        try:
            # Ensure the raw data files exists
            if not self.raw_path.exists():
                raise FileNotFoundError(f"Raw data not found at {self.raw_path}")
            
            df = pd.read_csv(self.raw_path)

            self.train_path.parent.mkdir(parents=True, exist_ok=True)

            # Split data 
            train_set, test_set = train_test_split(
                df,
                test_size = self.c["TEST_SIZE"],
                random_state = self.c["RANDOM_STATE"]
            )

            # Save the split data
            train_set.to_csv(self.train_path, index=False, header=True)
            test_set.to_csv(self.test_path, index=False, header=True)

            print(f"Data split and saved to {self.train_path.parent}")
            return str(self.train_path), str(self.test_path)
        
        except Exception as e:
            print(f"Error during at ingestion: {e}")
            raise e
