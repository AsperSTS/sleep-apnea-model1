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
def guardar_modelo(modelo, nombre, directorio=MODELS_DIR):
    """
    Guarda el modelo entrenado para su uso posterior.
    """
    # Crear directorio si no existe
    if not os.path.exists(directorio):
        os.makedirs(directorio)
    
    # Ruta del archivo
    ruta_modelo = os.path.join(directorio, f"{nombre}")
    
    # Guardar modelo
    joblib.dump(modelo, ruta_modelo)
    print(f"Modelo guardado en: {ruta_modelo}")
    
    return ruta_modelo

def cargar_modelo(ruta_modelo):
    """
    Carga un modelo guardado para hacer predicciones.
    """
    print(f"Cargando modelo desde {ruta_modelo}...")
    return joblib.load(ruta_modelo)

def visualizar_resultados(resultados, X_test, y_test):
    """
    Visualiza los resultados de los modelos.
    """
    # Create directory for visualizations
    if not os.path.exists('visualizaciones'):
        os.makedirs('visualizaciones')
    
    # Visualize model metrics
    plt.figure(figsize=(10, 6))
    modelos = list(resultados.keys())
    roc_auc_scores = [resultados[modelo]['roc_auc'] for modelo in modelos]
    
    sns.barplot(x=modelos, y=roc_auc_scores)
    plt.title('Comparación de ROC AUC por modelo')
    plt.ylabel('ROC AUC')
    plt.savefig('visualizaciones/comparacion_modelos.png')
    
    # Visualize confusion matrix for the best model
    mejor_modelo_nombre = max(resultados, key=lambda x: resultados[x]['roc_auc'])
    mejor_modelo = resultados[mejor_modelo_nombre]['modelo']
    
    plt.figure(figsize=(8, 6))
    matriz_confusion = resultados[mejor_modelo_nombre]['matriz_confusion']
    sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Apnea', 'Apnea'],
                yticklabels=['No Apnea', 'Apnea'])
    plt.title(f'Matriz de Confusión - {mejor_modelo_nombre}')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción')
    plt.savefig(f'visualizaciones/matriz_confusion_{mejor_modelo_nombre}.png')
    
    # Visualize feature importance
    if hasattr(mejor_modelo[-1], 'feature_importances_'):
        # Get preprocessor and feature names
        preprocessor = mejor_modelo.named_steps['preprocessor']
        
        # Initialize empty feature names list
        feature_names = []
        
        # Check for numerical features
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(columns)
            elif name == 'cat':
                # Get one-hot encoded feature names
                cat_cols = transformer.named_steps['onehot'].get_feature_names_out(columns)
                feature_names.extend(cat_cols)
        
        # If no transformers are found (passthrough case)
        if not feature_names:
            feature_names = X_test.columns.tolist()
            
        feature_names = np.array(feature_names)
        
        # Get feature importances
        importances = mejor_modelo[-1].feature_importances_
        
        # Make sure feature_names and importances have the same length
        if len(feature_names) == len(importances):
            indices = np.argsort(importances)[::-1]
            
            # Limit to the 15 most important features
            n_features = min(15, len(feature_names))
            
            plt.figure(figsize=(12, 8))
            plt.title('Importancia de Características')
            plt.bar(range(n_features), importances[indices[:n_features]], align='center')
            plt.xticks(range(n_features), feature_names[indices[:n_features]], rotation=90)
            plt.tight_layout()
            plt.savefig('visualizaciones/importancia_caracteristicas.png')
        else:
            print(f"Advertencia: Los nombres de características ({len(feature_names)}) no coinciden con las importancias ({len(importances)})")
    
    return mejor_modelo_nombre
# def visualizar_resultados(resultados, X_test, y_test):
#     """
#     Visualiza los resultados de los modelos.
#     """
#     # Crear directorio para visualizaciones
#     if not os.path.exists('visualizaciones'):
#         os.makedirs('visualizaciones')
    
#     # Visualizar métricas de los modelos
#     plt.figure(figsize=(10, 6))
#     modelos = list(resultados.keys())
#     roc_auc_scores = [resultados[modelo]['roc_auc'] for modelo in modelos]
    
#     sns.barplot(x=modelos, y=roc_auc_scores)
#     plt.title('Comparación de ROC AUC por modelo')
#     plt.ylabel('ROC AUC')
#     plt.savefig('visualizaciones/comparacion_modelos.png')
    
#     # Visualizar matriz de confusión del mejor modelo
#     mejor_modelo_nombre = max(resultados, key=lambda x: resultados[x]['roc_auc'])
#     mejor_modelo = resultados[mejor_modelo_nombre]['modelo']
    
#     plt.figure(figsize=(8, 6))
#     matriz_confusion = resultados[mejor_modelo_nombre]['matriz_confusion']
#     sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['No Apnea', 'Apnea'],
#                 yticklabels=['No Apnea', 'Apnea'])
#     plt.title(f'Matriz de Confusión - {mejor_modelo_nombre}')
#     plt.ylabel('Valor Real')
#     plt.xlabel('Predicción')
#     plt.savefig(f'visualizaciones/matriz_confusion_{mejor_modelo_nombre}.png')
    
#     # Visualizar importancia de características
#     if hasattr(mejor_modelo[-1], 'feature_importances_'):
#         # Obtener nombres de características post-transformación
#         preprocessor = mejor_modelo.named_steps['preprocessor']
#         cat_cols = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(CARACTERISTICAS_CATEGORICAS)
#         feature_names = np.concatenate([CARACTERISTICAS_NUMERICAS, cat_cols])
        
#         # Obtener importancia de características
#         importances = mejor_modelo[-1].feature_importances_
#         indices = np.argsort(importances)[::-1]
        
#         # Limitar a las 15 características más importantes
#         n_features = min(15, len(feature_names))
        
#         plt.figure(figsize=(12, 8))
#         plt.title('Importancia de Características')
#         plt.bar(range(n_features), importances[indices[:n_features]], align='center')
#         plt.xticks(range(n_features), feature_names[indices[:n_features]], rotation=90)
#         plt.tight_layout()
#         plt.savefig('visualizaciones/importancia_caracteristicas.png')
    
#     return mejor_modelo_nombre