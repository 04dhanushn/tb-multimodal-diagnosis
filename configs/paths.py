"""
Centralized path configuration for the TB multimodal project.

Update paths locally as needed.
Do NOT hardcode paths inside model or training scripts.
"""

import os

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------- Data paths ----------------
IMAGE_DATASET_DIR = "<UPDATE_IMAGE_DATASET_PATH>"
TABULAR_CSV_PATH = "<UPDATE_TABULAR_CSV_PATH>"

# ---------------- Output paths ----------------
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

IMAGE_MODEL_WEIGHTS = os.path.join(OUTPUT_DIR, "efficientnetv2_tb.pth")
TABULAR_MODEL_WEIGHTS = os.path.join(OUTPUT_DIR, "saint_tabular_best.pth")
TABULAR_SCALER_PATH = os.path.join(OUTPUT_DIR, "tab_scaler.pkl")

# ---------------- Misc ----------------
RANDOM_SEED = 42
