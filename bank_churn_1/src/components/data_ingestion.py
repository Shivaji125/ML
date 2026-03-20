import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.paths import get_data_path, get_config_path
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


class DataIngestion:
    def __init__(self):
        self.config = load_config(get_config_path("paths_config.yaml"))

        self.raw_path = get_data_path(self.config["RAW_DIR"]) / self.config["RAW_DATA_FILENAME"]
        self.train_path = get_data_path(self.config["PROCESSED_DIR"]) / self.config["TRAIN_DATA_FILENAME"]
        self.test_path = get_data_path(self.config["PROCESSED_DIR"]) / self.config["TEST_DATA_FILENAME"]

    def initiate_data_ingestion(self) -> tuple[str, str]:
        """Loads raw data, splits into train/test, and saves to disk."""
        logger.info("Loading data from: %s", self.raw_path)

        if not self.raw_path.exists():
            raise FileNotFoundError(f"Raw data not found at {self.raw_path}")

        df = pd.read_csv(self.raw_path)
        logger.info("Loaded dataset with shape %s", df.shape)

        self.train_path.parent.mkdir(parents=True, exist_ok=True)

        # Stratified split to preserve class distribution (important for imbalanced data)
        target_col = self.config["TARGET_COLUMN"]
        train_set, test_set = train_test_split(
            df,
            test_size=self.config["TEST_SIZE"],
            random_state=self.config["RANDOM_STATE"],
            stratify=df[target_col],
        )

        train_set.to_csv(self.train_path, index=False, header=True)
        test_set.to_csv(self.test_path, index=False, header=True)

        logger.info(
            "Data split complete — train: %d rows, test: %d rows, saved to %s",
            len(train_set), len(test_set), self.train_path.parent,
        )
        return str(self.train_path), str(self.test_path)
