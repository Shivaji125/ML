import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


def load_config(config_file_path: Path) -> dict:
    """Loads configuration from a YAML file."""
    try:
        with open(config_file_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded from %s", config_file_path)
        return config
    except Exception as e:
        logger.critical("Could not load configuration file %s: %s", config_file_path, e)
        raise
