"""
Configuration settings for model training, preprocessing, and file paths.
"""
import os

# --- Dataset Configuration ---

# This dataset currently yields the best performance for binary classification
BINARY_DATA_PATH = "shhs_hchs_33_25_21_21.csv"

# Primary dataset path
DATA_PATH = "shhs_hchs_4022_4022.csv"

# Columns to drop during training (e.g., targets or sensitive attributes)
VARIABLES_TO_EXCLUDE = ['apnea_binary', 'apnea_category']

# --- Model Selection ---

# List of models to execute in the pipeline
MODELS_TO_TRAIN = ["RandomForest", "GradientBoosting"]  # SVM currently disabled


# --- Output Directories: Visualizations ---

VISUAL_EDA_DIR = "visual_eda"
VISUAL_PREPROCESSED_DIR = "visual_pre"
VISUAL_MODEL_DIR = "visual_model"
TRAIN_TEST_DATA_DIR = "train_test"


# --- Output Directories: Reports ---

REPORTS_BASE_DIR = "reports"
SVM_REPORTS_PATH = os.path.join(REPORTS_BASE_DIR, "svm")
GB_REPORTS_PATH = os.path.join(REPORTS_BASE_DIR, "gradient_boost")
RF_REPORTS_PATH = os.path.join(REPORTS_BASE_DIR, "random_forest")
EDA_REPORTS_PATH = os.path.join(REPORTS_BASE_DIR, "eda")
PREPROCESSING_REPORTS_PATH = os.path.join(REPORTS_BASE_DIR, "preprocessing")


# --- Output Directories: Model Artifacts ---

MODELS_DIR = "models"
SVM_MODELS_DIR = os.path.join(MODELS_DIR, "svm")
RF_MODELS_DIR = os.path.join(MODELS_DIR, "random_forest")
GB_MODELS_DIR = os.path.join(MODELS_DIR, "gradient_boosting")

# Filename prefixes for saved models
SVM_MODEL_FILENAME = "SVM"
RF_MODEL_FILENAME = "RandomForest"
GB_MODEL_FILENAME = "GradientBoosting"


# --- Preprocessing Parameters ---

# Age filtering thresholds
EDAD_MIN = 28
EDAD_MAX = 75

# Feature Selection
CARACTERISTICAS_NUMERICAS = [
    'nsrr_age', 'nsrr_bmi', 'nsrr_bp_systolic', 'nsrr_bp_diastolic'
]

CARACTERISTICAS_CATEGORICAS = [
    'nsrr_sex', 'nsrr_race', 'nsrr_ethnicity', 
    'nsrr_current_smoker', 'nsrr_ever_smoker'
]