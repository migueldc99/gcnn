"""Centralized path resolution for the GCNN project.

Override the data directory by setting the GCNN_DATA_DIR environment variable.
"""

import os
from pathlib import Path

# Project root: two levels up from this file (src/gcnn/paths.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Allow override via environment variable
_data_dir_override = os.environ.get("GCNN_DATA_DIR")

if _data_dir_override:
    DATA_DIR = Path(_data_dir_override)
else:
    DATA_DIR = PROJECT_ROOT / "data"

# Sub-directories within the data folder
ORIGINAL_DATASET = DATA_DIR / "original_dataset"
DATASET = DATA_DIR / "dataset"
TRAINING_SET = DATA_DIR / "training_set"
VALIDATION_SET = DATA_DIR / "validation_set"
TEST_SET = DATA_DIR / "test_set"
PROCESSED_DIR = DATA_DIR / "processed_graphs"

# Configuration and outputs
CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def ensure_data_dirs() -> None:
    """Create all data subdirectories if they don't exist."""
    for d in [ORIGINAL_DATASET, DATASET, TRAINING_SET, VALIDATION_SET, TEST_SET, PROCESSED_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def ensure_output_dir() -> None:
    """Create the outputs directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
