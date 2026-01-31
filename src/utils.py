"""
Utility functions for file handling, data loading, and visualization.
"""
import pandas as pd
import os
import joblib
import datetime
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# Config imports
from config import (
    MODELS_DIR, TRAIN_TEST_DATA_DIR, DATA_PATH, VARIABLES_TO_EXCLUDE, 
    VISUAL_EDA_DIR, VISUAL_PREPROCESSED_DIR, VISUAL_MODEL_DIR, 
    SVM_REPORTS_PATH, GB_REPORTS_PATH, RF_REPORTS_PATH, EDA_REPORTS_PATH,
    PREPROCESSING_REPORTS_PATH, SVM_MODELS_DIR, RF_MODELS_DIR, GB_MODELS_DIR
)

def load_data() -> pd.DataFrame:
    """
    Loads the dataset from CSV and removes excluded columns defined in config.
    """
    print(f"Cargando datos desde {DATA_PATH}...")
    
    datos = pd.read_csv(DATA_PATH)
    if VARIABLES_TO_EXCLUDE:
        try:
            datos = datos.drop(VARIABLES_TO_EXCLUDE, axis=1)
        except:
            print("Advertencia: No se pudieron borrar las columnas excluidas (verificar nombres).")
    return datos

def save_csv_list(nombre_columnas:str, nombre_archivo:str, lista: List): 
    """Saves a list to a CSV file."""
    df = pd.DataFrame(lista, columns=[nombre_columnas])
    df.to_csv(os.path.join(MODELS_DIR, nombre_archivo), index=False)   

def generate_unique_timestamp() -> str:
    """
    Generates a unique timestamp ID (seconds-day-month-year) to prevent file overwrites.
    """
    now = datetime.datetime.now()

    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    time_difference = now - start_of_day
    numeric_timestamp = int(time_difference.total_seconds())

    day = now.strftime("%d")   
    month = now.strftime("%m") 
    year = now.strftime("%Y")  

    timestamp = f"{numeric_timestamp}-{day}-{month}-{year}"
    return timestamp

def create_folders() -> bool:
    """
    Creates the necessary project directory structure if it doesn't exist.
    """
    folders_to_create = [
        VISUAL_EDA_DIR, VISUAL_PREPROCESSED_DIR, VISUAL_MODEL_DIR,
        TRAIN_TEST_DATA_DIR, SVM_REPORTS_PATH, GB_REPORTS_PATH,
        RF_REPORTS_PATH, EDA_REPORTS_PATH, PREPROCESSING_REPORTS_PATH,
        MODELS_DIR, SVM_MODELS_DIR, RF_MODELS_DIR, GB_MODELS_DIR
    ]

    print("Creando carpetas...")
    for folder in folders_to_create:
        try:
            os.makedirs(folder, exist_ok=True)
            print(f"  Carpeta lista: '{folder}'")
        except Exception as e:
            print(f"  ERROR: Falló al crear '{folder}': {e}")
            return False 

    print("\nCarpetas listas.")
    return True

def save_train_test(X_train, X_test, y_train, y_test, caracteristicas_seleccionadas):
    """
    Saves training and testing splits to CSV files for auditing.
    """
    if not os.path.exists(TRAIN_TEST_DATA_DIR):
        os.makedirs(TRAIN_TEST_DATA_DIR)

    pd.DataFrame(X_train, columns=caracteristicas_seleccionadas).to_csv(os.path.join(TRAIN_TEST_DATA_DIR, 'X_train.csv'), index=False)
    pd.DataFrame(X_test, columns=caracteristicas_seleccionadas).to_csv(os.path.join(TRAIN_TEST_DATA_DIR, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(TRAIN_TEST_DATA_DIR, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(TRAIN_TEST_DATA_DIR, 'y_test.csv'), index=False)

def save_model(model: dict, mode: str, filepath: str, model_name:str, 
               precision: float, recall: float, f1:float, roc_auc: float):
    """
    Persists the trained model to a .joblib file. 
    Includes performance metrics in the filename for quick reference.
    """
    print("\n Guardando modelo...")
    print(f"Precision: {int(precision*100)}%")
    print(f"Recall: {int(recall*100)}%")
    print(f"F1: {int(f1*100)}%")
    print(f"ROC-AUC: {int(roc_auc*100)}%")

    if model['trained_model'] is None:
        raise ValueError("Error: El modelo está vacío. Entrénalo antes de guardar.")

    full_name = f"{mode}_{model_name}_{int(precision*100)}_{int(recall*100)}_{int(f1*100)}_{int(roc_auc*100)}.joblib"
    
    joblib.dump(model, os.path.join(filepath, full_name))
    print(f"Modelo guardado en: {os.path.join(filepath, full_name)}")

def visualizar_resultados_modelo(resultados, nombre_modelo, tipo_clasificacion='binario', reports_path=None):
    """
    Generates performance plots and a summary report.
    
    Outputs:
    1. Metrics bar chart
    2. Confusion Matrix
    3. Probability distribution histogram
    4. Cross-validation stability boxplot
    5. Feature importance
    6. ROC and PR curves (Binary only)
    7. Text summary report
    """
    # Determine report path based on model name if not provided
    if reports_path is None:
        model_paths = {
            'RandomForest': RF_REPORTS_PATH,
            'SVM': SVM_REPORTS_PATH, 
            'GradientBoosting': GB_REPORTS_PATH
        }
        reports_path = model_paths.get(nombre_modelo, MODELS_DIR)

    if not os.path.exists(reports_path):
        os.makedirs(reports_path)
    
    y_test, y_prob = resultados['test_data']
    
    #  1. Metrics Chart 
    plt.figure(figsize=(12, 6))
    
    if tipo_clasificacion == 'binario':
        metricas = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    else:
        metricas = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    metricas_disponibles = [m for m in metricas if m in resultados]
    valores = [resultados[m] for m in metricas_disponibles]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    bars = plt.bar(metricas_disponibles, valores, color=colors[:len(valores)])
    plt.title(f'Métricas: {nombre_modelo} ({tipo_clasificacion})', fontsize=14, fontweight='bold')
    plt.ylim(0, 1)
    
    for bar, v in zip(bars, valores):
        plt.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{reports_path}/{tipo_clasificacion}_metricas_{nombre_modelo}.png', dpi=300)
    plt.close()
    
    #  2. Confusion Matrix 
    
    plt.figure(figsize=(8, 6))
    cm = resultados['confusion_matrix']
    clases = np.unique(y_test)
    
    # Normalize
    cm_normalized = cm.astype('float') / np.nan_to_num(cm.sum(axis=1, keepdims=True))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=clases)
    disp.plot(cmap='Blues', values_format='.2%', ax=plt.gca())
    plt.title(f'Matriz de Confusión - {nombre_modelo}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{reports_path}/{tipo_clasificacion}_matriz_confusion_{nombre_modelo}.png', dpi=300)
    plt.close()
    
    #  3. Probability Distribution 
    if tipo_clasificacion == 'binario':
        plt.figure(figsize=(12, 6))
        for clase in np.unique(y_test):
            mask = y_test == clase
            if np.any(mask):
                if y_prob.ndim == 1:
                    prob_data = y_prob[mask]
                else:
                    prob_data = y_prob[mask, 1] if y_prob.shape[1] > 1 else y_prob[mask, 0]
                
                plt.hist(prob_data, bins=30, alpha=0.7, label=f'Clase real {clase}', density=True)
        
        plt.title('¿Qué tan seguro estaba el modelo?')
        plt.xlabel('Probabilidad asignada')
        plt.legend()
        plt.savefig(f'{reports_path}/{tipo_clasificacion}_distribucion_probabilidades_{nombre_modelo}.png', dpi=300)
        plt.close()
        
    else: # Multiclass
        plt.figure(figsize=(15, 10))
        n_clases = y_prob.shape[1]
        for i in range(n_clases):
            plt.subplot(1, n_clases, i+1)
            for j in range(n_clases):
                mask = y_test == j
                if np.any(mask):
                    sns.kdeplot(y_prob[mask, i], label=f'Real {j}', alpha=0.7)
            plt.title(f'Probabilidad predicha para Clase {i}')
            plt.legend()
        plt.tight_layout()
        plt.savefig(f'{reports_path}/{tipo_clasificacion}_distribucion_probabilidades_{nombre_modelo}.png', dpi=300)
        plt.close()
    
    #  4. Cross-Validation Stability 
    plt.figure(figsize=(10, 6))
    cv_scores = resultados['cv_scores']
    plt.boxplot(cv_scores, patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.7))
    plt.title(f'Estabilidad del modelo (CV Scores) - {nombre_modelo}')
    plt.axhline(y=np.mean(cv_scores), color='r', linestyle='--', label=f'Media: {np.mean(cv_scores):.3f}')
    plt.legend()
    plt.savefig(f'{reports_path}/{tipo_clasificacion}_cv_scores_{nombre_modelo}.png', dpi=300)
    plt.close()
    
    #  5. Feature Importance 
    if 'feature_importance' in resultados:
        plt.figure(figsize=(12, 8))
        importances = resultados['feature_importance']
        indices = np.argsort(importances)[::-1]
        top_n = min(15, len(importances))
        
        plt.bar(range(top_n), importances[indices[:top_n]])
        plt.title(f'Variables más importantes - {nombre_modelo}')
        plt.xticks(range(top_n), [f'Var_{indices[i]}' for i in range(top_n)], rotation=45)
        plt.tight_layout()
        plt.savefig(f'{reports_path}/{tipo_clasificacion}_importancia_{nombre_modelo}.png', dpi=300)
        plt.close()
    
    #  6. ROC and PR Curves (Binary only) 
    if tipo_clasificacion == 'binario' and 'roc_curve' in resultados:
        plt.figure(figsize=(15, 5))
        
        # ROC
        plt.subplot(1, 2, 1)
        fpr, tpr = resultados['roc_curve']
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {resultados["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
        plt.title('Curva ROC')
        plt.legend()
        
        # PR
        if 'pr_curve' in resultados:
            plt.subplot(1, 2, 2)
            precision, recall = resultados['pr_curve']
            plt.plot(recall, precision, color='blue', lw=2, label=f'PR (AUC = {resultados["pr_auc"]:.3f})')
            plt.title('Curva Precision-Recall')
            plt.legend()
        
        plt.savefig(f'{reports_path}/{tipo_clasificacion}_curvas_{nombre_modelo}.png', dpi=300)
        plt.close()
    
    #  7. Save Text Report 
    with open(f'{reports_path}/{tipo_clasificacion}_reporte_{nombre_modelo}.txt', 'w', encoding='utf-8') as f:
        f.write(f"===== REPORTE: {nombre_modelo} ({tipo_clasificacion.upper()}) =====\n\n")
        f.write(f"Accuracy: {resultados['accuracy']:.4f}\n")
        f.write(f"Precision: {resultados['precision']:.4f}\n")
        f.write(f"Recall: {resultados['recall']:.4f}\n")
        f.write(f"F1 Score: {resultados['f1']:.4f}\n")
        f.write(f"ROC AUC: {resultados['roc_auc']:.4f}\n")
        
        f.write("\n===== DETALLE POR CLASE =====\n")
        f.write(resultados['classification_report'])
        
        f.write("\n===== VALIDACIÓN CRUZADA (Estabilidad) =====\n")
        f.write(f"Media: {np.mean(resultados['cv_scores']):.4f}\n")
        f.write(f"Mínimo: {np.min(resultados['cv_scores']):.4f} / Máximo: {np.max(resultados['cv_scores']):.4f}\n")
        
        if 'feature_importance' in resultados:
            f.write("\n===== TOP 10 VARIABLES MÁS IMPORTANTES =====\n")
            importances = resultados['feature_importance']
            indices = np.argsort(importances)[::-1]
            for i in range(min(10, len(importances))):
                f.write(f"{i+1}. Variable {indices[i]}: {importances[indices[i]]:.4f}\n")
    
    print(f"Gráficos y reporte guardados en: {reports_path}/")
    return nombre_modelo

def cargar_modelo(ruta_modelo):
    """Loads a .joblib model from a specific path."""
    print(f"Cargando modelo: {ruta_modelo}...")
    return joblib.load(ruta_modelo)

def load_model(models_list: str) -> List:
    """
    Loads multiple models (RF, SVM, GB) from their respective directories based on filenames.
    """
    models = []
    try:
        for model_name in models_list:
            if "RandomForest" in model_name:
                print(f"Cargando RF: {model_name}")
                models.append(joblib.load(os.path.join(RF_MODELS_DIR, model_name)))
            elif "SVM" in model_name:
                print(f"Cargando SVM: {model_name}")
                models.append(joblib.load(os.path.join(SVM_MODELS_DIR, model_name)))
            elif "GradientBoosting" in model_name:
                print(f"Cargando GB: {model_name}")
                models.append(joblib.load(os.path.join(GB_MODELS_DIR, model_name)))
        return models
    except Exception as e:
        print(f"Error cargando modelos: {e}")
        return []