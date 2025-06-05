"""
Utilidades y funciones auxiliares para el procesamiento de datos
"""
import pandas as pd
import os
import joblib
import os
import datetime
from typing import List
from config import MODELS_DIR, TRAIN_TEST_DATA_DIR, DATA_PATH, VARIABLES_TO_EXCLUDE, VISUAL_EDA_DIR , VISUAL_PREPROCESSED_DIR 
from config import VISUAL_MODEL_DIR, TRAIN_TEST_DATA_DIR ,SVM_REPORTS_PATH,GB_REPORTS_PATH,RF_REPORTS_PATH,EDA_REPORTS_PATH
from config import PREPROCESSING_REPORTS_PATH,MODELS_DIR,SVM_MODELS_DIR,RF_MODELS_DIR,GB_MODELS_DIR   

def load_data() -> pd.DataFrame:
    """
    Carga los datos desde un archivo CSV.
    """
    print(f"Cargando datos desde {DATA_PATH}...")
    
    datos = pd.read_csv(DATA_PATH)
    if VARIABLES_TO_EXCLUDE:
        datos = datos.drop(VARIABLES_TO_EXCLUDE, axis=1)
    return datos

def save_csv_list(nombre_columnas:str, nombre_archivo:str, lista: List): 
    df = pd.DataFrame(lista, columns=[nombre_columnas])
    df.to_csv(os.path.join(MODELS_DIR,nombre_archivo), index=False)   

def generate_unique_timestamp() -> str:
    """
    Genera un timestamp con el formato 'segundoexactodeldia-dia-mes-anio'.
    """
    now = datetime.datetime.now()

    # Calcular los segundos exactos desde la medianoche de hoy
    # 1. Obtener el inicio del día (medianoche de la fecha actual)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    # 2. Calcular la diferencia de tiempo entre ahora y el inicio del día
    time_difference = now - start_of_day
    # 3. Convertir la diferencia a segundos totales (y redondear a entero)
    numeric_timestamp = int(time_difference.total_seconds())

    day = now.strftime("%d")   # Día con dos dígitos (ej. 05)
    month = now.strftime("%m") # Mes con dos dígitos (ej. 06)
    year = now.strftime("%Y")  # Año con cuatro dígitos (ej. 2025)

    # Combina todo en el formato deseado
    timestamp = f"{numeric_timestamp}-{day}-{month}-{year}"
    return timestamp

def create_folders() -> bool:
    """
    Crea una estructura de carpetas predefinida para organizar resultados,
    reportes y modelos de un proyecto de Machine Learning.

    Returns:
        bool: True si todas las carpetas se crearon o ya existían; False si ocurrió un error.
    """
    # Lista de todas las carpetas a crear
    folders_to_create = [
        VISUAL_EDA_DIR,
        VISUAL_PREPROCESSED_DIR,
        VISUAL_MODEL_DIR,
        TRAIN_TEST_DATA_DIR,
        SVM_REPORTS_PATH,
        GB_REPORTS_PATH,
        RF_REPORTS_PATH,
        EDA_REPORTS_PATH,
        PREPROCESSING_REPORTS_PATH,
        MODELS_DIR, # Aunque sus subcarpetas se crearán, la base siempre es útil
        SVM_MODELS_DIR,
        RF_MODELS_DIR,
        GB_MODELS_DIR
    ]

    print("Intentando crear la estructura de carpetas...")
    for folder in folders_to_create:
        try:
            # Crea todos los directorios intermedios si no existen.
            os.makedirs(folder, exist_ok=True) # Lanza FileExistsError si la carpeta ya existe.
            print(f"  Carpeta '{folder}' creada/existente.")
        except Exception as e:
            
            print(f"  ERROR: No se pudo crear la carpeta '{folder}': {e}")
            return False # Retorna False si falla la creación de alguna carpeta

    print("\nProceso de creación de carpetas completado.")
    return True # Retorna True si todas las carpetas se crearon o ya existían

# --- Ejemplo de uso ---
if __name__ == "__main__":
    if create_folders():
        print("¡Todas las carpetas necesarias están listas para tu proyecto!")
    else:
        print("Hubo un problema al crear algunas carpetas. Revisa los mensajes de error.")

    # Puedes limpiar las carpetas para probar de nuevo si quieres
    # import shutil
    # if os.path.exists("results"): shutil.rmtree("results")
    # if os.path.exists("reports"): shutil.rmtree("reports")
    # if os.path.exists("data"): shutil.rmtree("data")
    # if os.path.exists("models"): shutil.rmtree("models")

def save_train_test(X_train, X_test, y_train, y_test, caracteristicas_seleccionadas):

    # 4. Crear la carpeta si no existe
    if not os.path.exists(TRAIN_TEST_DATA_DIR):
        os.makedirs(TRAIN_TEST_DATA_DIR)

    # 5. Guardar los conjuntos en archivos CSV
    pd.DataFrame(X_train, columns=caracteristicas_seleccionadas).to_csv(os.path.join(TRAIN_TEST_DATA_DIR, 'X_train.csv'), index=False)
    pd.DataFrame(X_test, columns=caracteristicas_seleccionadas).to_csv(os.path.join(TRAIN_TEST_DATA_DIR, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(TRAIN_TEST_DATA_DIR, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(TRAIN_TEST_DATA_DIR, 'y_test.csv'), index=False)

def cargar_modelo(ruta_modelo):
    """
    Carga un modelo guardado para hacer predicciones.
    """
    print(f"Cargando modelo desde {ruta_modelo}...")
    return joblib.load(ruta_modelo)

def load_model_SVM(self, filepath):
    """
    Carga un modelo SVM desde un archivo.
    
    Args:
        filepath: Ruta del archivo del modelo
    """
    import joblib
    
    # Cargar modelo
    self.svm_classifier = joblib.load(filepath)
    print(f"Modelo cargado desde {filepath}")
    
    # Extraer parámetros del modelo cargado
    self.svm_c_parameter = self.svm_classifier.C
    self.svm_kernel_parameter = self.svm_classifier.kernel
    self.svm_gamma_parameter = self.svm_classifier.gamma
    self.svm_class_weight_parameter = self.svm_classifier.class_weight

