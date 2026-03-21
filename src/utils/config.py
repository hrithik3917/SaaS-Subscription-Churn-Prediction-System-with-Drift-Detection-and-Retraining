"""
config.py - Central Configuration Loader
=========================================
Provides file paths and params.yaml access.
Database connection is handled by db.py.
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class Config:
    """All project configuration in one class."""
    
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
    INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR = PROJECT_ROOT / "models"
    PARAMS_PATH = PROJECT_ROOT / "configs" / "params.yaml"
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    
    @classmethod
    def load_params(cls):
        """Load and return the params.yaml configuration."""
        with open(cls.PARAMS_PATH, "r") as f:
            return yaml.safe_load(f)


if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data dir: {Config.RAW_DATA_DIR}")
    params = Config.load_params()
    print(f"Model type: {params['model']['type']}")