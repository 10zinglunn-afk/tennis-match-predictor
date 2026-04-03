"""
Configuration file for Tennis Prediction Project

This module contains all configuration constants and parameters used throughout
the project for data processing, feature engineering, and model training.
"""

# Data paths
DATA_RAW_PATH = "data/raw"
DATA_PROCESSED_PATH = "data/processed"
MODELS_PATH = "models"

# Data source configuration
JEFF_SACKMANN_BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp"
TOURNAMENTS_FILENAME = "atp_matches_{year}.csv"
RANKINGS_FILENAME = "atp_rankings_{year}.csv"

# Feature engineering parameters
ELO_INITIAL_RATING = 1500
ELO_K_FACTOR = 32
FORM_WINDOW_DAYS = 90  # Days to look back for form calculation
FATIGUE_WEIGHT = 0.5  # Weight for fatigue factor in calculations

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_NAME = "tennis_prediction_model"

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
