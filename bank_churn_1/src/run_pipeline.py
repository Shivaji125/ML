import logging
import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_trainer import ModelTrainer
from src.utils.config_loader import load_config
from src.utils.paths import get_config_path

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging for the entire pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-35s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_pipeline() -> dict:
    """Executes the end-to-end modular ML pipeline."""
    logger.info("Starting end-to-end ML pipeline execution.")

    config = load_config(get_config_path("paths_config.yaml"))

    # Step 1: Data Ingestion
    logger.info("Step 1/4 — Data Ingestion")
    ingestor = DataIngestion()
    train_path, test_path = ingestor.initiate_data_ingestion()

    # Step 2: Data Validation
    logger.info("Step 2/4 — Data Validation")
    validator = DataValidation(config)
    is_data_valid = validator.initiate_data_validation()

    if not is_data_valid:
        logger.error("Pipeline halted — critical data quality issues detected.")
        sys.exit(1)

    # Step 3: Data Transformation
    logger.info("Step 3/4 — Data Transformation")
    transformer = DataTransformation()
    X_train, y_train, X_test, y_test, preprocessor_path = (
        transformer.initiate_data_transformation(train_path, test_path)
    )

    # Step 4: Model Training
    logger.info("Step 4/4 — Model Training")
    trainer = ModelTrainer()
    metrics = trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)

    logger.info("Pipeline completed successfully.")
    return metrics


if __name__ == "__main__":
    setup_logging()
    try:
        run_pipeline()
    except Exception:
        logger.exception("Pipeline FAILED")
        sys.exit(1)

# Run from project root: python -m src.run_pipeline
