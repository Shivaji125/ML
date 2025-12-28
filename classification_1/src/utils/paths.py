from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# -- Path Getters ---
def get_root() -> Path:
    """Returns the absolute path of the project root."""
    return PROJECT_ROOT

def get_data_path(subdirectory:str = "processed") -> Path:
    """Returns the absolute path to a subdirectory within the data folder."""
    return PROJECT_ROOT / "data" / subdirectory

def get_config_path(config_file:str = "paths_config.yaml") -> Path:
    """Returns the absolute path to a configuration file."""
    return PROJECT_ROOT / "config" / config_file

def get_model_path() -> Path:
    """Returns the absolute path to the models folder."""
    return PROJECT_ROOT / "models"