"""Configuration parameters for the project"""

# Data paths
DATA_PATH = "data/train.parquet"
VAL_PATH = "data/validation.parquet"
META_PATH = "data/features.json"
META_MODEL = "data/meta_model.parquet"
DATA_VERSION = "v1.0"

# Model parameters
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10

# Feature sets
FEATURE_SETS = {
    "small": 42,
    "medium": 705,
    "all": 2376
}

# Training parameters
EARLY_STOPPING_PATIENCE = 8
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
MIN_LR = 0.00001