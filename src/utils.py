"""
Utilidades y funciones auxiliares para el procesamiento de datos
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os
import joblib
from config import *
from typing import List

def cargar_datos(ruta_archivo):
    """
    Carga los datos desde un archivo CSV.
    """
    print(f"Cargando datos desde {ruta_archivo}...")
    return pd.read_csv(ruta_archivo)
def guardar_lista_csv(nombre_columnas:str, nombre_archivo:str, lista: List): 
    df = pd.DataFrame(lista, columns=[nombre_columnas])
    df.to_csv(os.path.join(MODELS_DIR,nombre_archivo), index=False)   
    
def guardar_train_test(X_train, X_test, y_train, y_test, caracteristicas_seleccionadas):

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
