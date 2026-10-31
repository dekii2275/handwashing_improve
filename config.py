"""
Configuration settings for Hand-Washing Movement Recognition
Based on Hand Skeleton Landmarks (Ivanovs & Ivanova, Baltic DB&IS 2026)
"""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
LANDMARKS_DIR = DATA_DIR / "landmarks"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

for d in [
    DATA_DIR,
    RAW_DIR,
    LANDMARKS_DIR,
    PROCESSED_DIR,
    CHECKPOINTS_DIR,
    RESULTS_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset Zenodo Metadata
ZENODO_RECORD_ID = "5808789"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
ZENODO_FILES = {
    "summary": "summary.csv",
    "statistics": "statistics.csv",
    "interface_1": "Interface_number_1.zip",
    "interface_2": "Interface_number_2.zip",
    "interface_3": "Interface_number_3.zip",
}

# 7 Hand-washing classes (WHO steps + class 0 other)
CLASS_NAMES = {
    0: "Other / non-WHO",
    1: "Palm to palm",
    2: "Palm over dorsum, fingers interlaced",
    3: "Palm to palm, fingers interlaced",
    4: "Backs of fingers to opposing palm",
    5: "Rotational rubbing of thumb",
    6: "Fingertips to palm",
}
NUM_CLASSES = 7

# Split configuration
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20
RANDOM_SEED = 42

# MediaPipe configuration
NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3
RAW_FEATURE_DIM = NUM_HANDS * LANDMARKS_PER_HAND * COORDS_PER_LANDMARK  # 126

# Training configuration
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_EPOCHS = 100
DROPOUT_RATE = 0.2

# Best reported configurations from the paper
BEST_CONFIGS = {
    "temporal_cnn": {
        "norm_type": "norm_loc",
        "impute": True,
        "feature_type": "raw_landmarks",
        "window_size": 48,
        "filters": 64,
        "kernel_size": 3,
        "num_layers": 2,
        "reported_f1": 70.5,
        "reported_acc": 70.3,
    },
    "gru": {
        "norm_type": "norm_zoom",
        "impute": False,
        "feature_type": "raw_landmarks",
        "window_size": 48,
        "hidden_size": 128,
        "num_layers": 2,
        "reported_f1": 68.1,
        "reported_acc": 69.7,
    },
    "lstm": {
        "norm_type": "none",
        "impute": True,
        "feature_type": "raw_landmarks",
        "window_size": 30,
        "hidden_size": 64,
        "num_layers": 2,
        "reported_f1": 65.7,
        "reported_acc": 65.9,
    },
}
