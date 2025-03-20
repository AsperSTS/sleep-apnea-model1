"""
Archivo de configuración con constantes y parámetros del modelo
"""

# Rutas de archivos
DATA_PATH = "shhs-harmonized-filtered.csv"
MODELS_DIR = "modelos"
MODEL_FILENAME = "modelo_apnea_obstructiva.joblib"

# Parámetros para preprocesamiento
EDAD_MIN = 18
EDAD_MAX = 60

# Características para el modelo
CARACTERISTICAS_NUMERICAS = [
    'nsrr_age', 'nsrr_bmi', 'nsrr_bp_systolic', 'nsrr_bp_diastolic', 
    'nsrr_phrnumar_f1'
]

CARACTERISTICAS_CATEGORICAS = [
    'nsrr_sex', 'nsrr_race', 'nsrr_ethnicity', 
    'nsrr_current_smoker', 'nsrr_ever_smoker'
]

# Parámetros para los modelos
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Parámetros para GridSearchCV
PARAMS_RANDOM_FOREST = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

PARAMS_GRADIENT_BOOSTING = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.1, 0.05],
    'classifier__max_depth': [3, 5]
}