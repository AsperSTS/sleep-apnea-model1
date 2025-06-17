"""
Archivo de configuración con constantes y parámetros del modelo
"""
import os

# Informacion de los datasets
BINARY_DATA_PATH = "shhs_hchs_33_25_21_21.csv" # Este dataset da mejores resultados con datos binarios

DATA_PATH = "shhs_hchs_33_25_21_21.csv" # "shhs_hchs.csv"
VARIABLES_TO_EXCLUDE = ['nsrr_ethnicity', 'nsrr_race']

# Modelos a entrenar -> ["SVM", "RandomForest","GradientBoosting"]
MODELS_TO_TRAIN = ["RandomForest","GradientBoosting","SVM"]

# Carpetas de resultados
VISUAL_EDA_DIR = "visual_eda"
VISUAL_PREPROCESSED_DIR = "visual_pre"
VISUAL_MODEL_DIR = "visual_model"
TRAIN_TEST_DATA_DIR = "train_test"

# Rutas de reportes
REPORTS_BASE_DIR = "reports"
SVM_REPORTS_PATH = os.path.join(REPORTS_BASE_DIR, "svm")
GB_REPORTS_PATH = os.path.join(REPORTS_BASE_DIR, "gradient_boost")
RF_REPORTS_PATH = os.path.join(REPORTS_BASE_DIR, "random_forest")
EDA_REPORTS_PATH = os.path.join(REPORTS_BASE_DIR, "eda")
PREPROCESSING_REPORTS_PATH = os.path.join(REPORTS_BASE_DIR, "preprocessing")

# Carpetas de modelos
MODELS_DIR = "models"
SVM_MODELS_DIR = os.path.join(MODELS_DIR, "svm")
RF_MODELS_DIR = os.path.join(MODELS_DIR, "random_forest")
GB_MODELS_DIR = os.path.join(MODELS_DIR, "gradient_boosting")


# Nombres de modelos a almacenar
SVM_MODEL_FILENAME = "SVM"
RF_MODEL_FILENAME = "RandomForest"
GB_MODEL_FILENAME = "GradientBoosting"


# Parámetros para preprocesamiento
EDAD_MIN = 28
EDAD_MAX = 75

# Características para el modelo
CARACTERISTICAS_NUMERICAS = [
    'nsrr_age', 'nsrr_bmi', 'nsrr_bp_systolic', 'nsrr_bp_diastolic'#, 
    #'nsrr_phrnumar_f1'
]

CARACTERISTICAS_CATEGORICAS = [
    'nsrr_sex', 'nsrr_race', 'nsrr_ethnicity', 
    'nsrr_current_smoker', 'nsrr_ever_smoker'
]




# Parámetros para los modelos
