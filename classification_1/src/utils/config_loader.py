import yaml
from pathlib import Path

def load_config(config_file_path: Path) -> dict:
    """Loads configuration form a YAML file."""
    try:
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"FATAL ERROR: Could not load configuration file {config_file_path}: {e}")
        raise e