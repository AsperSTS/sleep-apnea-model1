"""
Utilidades y funciones auxiliares para el procesamiento de datos
"""
import pandas as pd
import os
import joblib
import os
import datetime
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import ConfusionMatrixDisplay

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

def save_train_test(X_train, X_test, y_train, y_test, caracteristicas_seleccionadas):

    # 4. Crear la carpeta si no existe
    if not os.path.exists(TRAIN_TEST_DATA_DIR):
        os.makedirs(TRAIN_TEST_DATA_DIR)

    # 5. Guardar los conjuntos en archivos CSV
    pd.DataFrame(X_train, columns=caracteristicas_seleccionadas).to_csv(os.path.join(TRAIN_TEST_DATA_DIR, 'X_train.csv'), index=False)
    pd.DataFrame(X_test, columns=caracteristicas_seleccionadas).to_csv(os.path.join(TRAIN_TEST_DATA_DIR, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(TRAIN_TEST_DATA_DIR, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(TRAIN_TEST_DATA_DIR, 'y_test.csv'), index=False)

def save_model(model: dict, mode: str, filepath: str, model_name:str, 
               precision: float, recall: float, f1:float, roc_auc: float):
    """
    Guarda el modelo SVM entrenado en un archivo.
    
    Args:
        model: Diccionario que contiene los siguientes parametros
            @param features Nombres de las caracteristicas en el modelo
            @param trained_model Modelo entrenado
            @param model_name Nombre del modelo, 'RandomForest', 'SVM' o 'GradientBoosting'
            @param 
            
        mode: Modo de clasificacion, 'binario' o 'multiclase'
        filepath: Directorio para almacenar el modelo
        model_name: Nombre del modelo o algoritmo - "SVM", "RandomForest", "GradientBoosting"
        precision: Precicion en validacion
        recall: Recall en validacion
        f1: F1-score en validacion
        roc_auc:ROC-AUC en validacion    
    
    """
    print("\n Guardando modelo...")
    print(f"Precision: {precision} : {int(precision*100)}%")
    print(f"Recall: {recall} : {int(recall*100)}%")
    print(f"F1: {f1} : {int(f1*100)}%")
    print(f"ROC-AUC: {roc_auc} : {int(roc_auc*100)}%")
    # Verificar que el modelo esté entrenado
    if model['trained_model'] is None:
        raise ValueError("El modelo SVM no ha sido entrenado. Llame a train_svm() primero.")
    model_name = mode+"_"+model_name+"_"+str(int(precision*100))+"_"+str(int(recall*100))+"_"+str(int(f1*100))+"_"+str(int(roc_auc*100))+".joblib"
    # Guardar modelo
    joblib.dump(model, os.path.join(filepath,model_name) )
    print(f"Modelo guardado en {os.path.join(filepath,model_name)}")

def visualizar_resultados_modelo(resultados, nombre_modelo, tipo_clasificacion='binario', reports_path=None):
    """
    Visualiza los resultados de modelos de Machine Learning (RandomForest, SVM, GradientBoosting) 
    tanto para clasificación binario como multiclase.
    
    Args:
        resultados: Diccionario con los resultados del entrenamiento
        nombre_modelo: Nombre del modelo para usar en los archivos guardados (ej: 'RandomForest', 'SVM_Binario', 'GradientBoosting')
        tipo_clasificacion: 'binario' o 'multiclase' para adaptar las visualizaciones
        reports_path: Ruta donde guardar los reportes. Si es None, se usará el nombre del modelo
    """
    
    
    # Determinar la ruta de reportes
    if reports_path is None:
        # Mapeo de nombres de modelo a rutas por defecto
        model_paths = {
            'RandomForest': RF_REPORTS_PATH,
            'SVM': SVM_REPORTS_PATH, 
            'GradientBoosting': GB_REPORTS_PATH
        }

        reports_path = model_paths[nombre_modelo]
    # Crear directorio para visualizaciones si no existe
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)
    
    # Obtener datos de prueba
    y_test,  y_prob = resultados['test_data']
    
    # 1. Métricas del modelo en gráfico de barras
    plt.figure(figsize=(12, 6))
    
    if tipo_clasificacion == 'binario':
        metricas = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        valores = [resultados[metrica] for metrica in metricas if metrica in resultados]
        metricas_disponibles = [metrica for metrica in metricas if metrica in resultados]
    else:
        metricas = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        valores = [resultados[metrica] for metrica in metricas if metrica in resultados]
        metricas_disponibles = [metrica for metrica in metricas if metrica in resultados]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'][:len(valores)]
    bars = plt.bar(metricas_disponibles, valores, color=colors)
    plt.title(f'Métricas del modelo {nombre_modelo} ({tipo_clasificacion.capitalize()})', 
              fontsize=14, fontweight='bold')
    plt.ylim(0, 1)
    plt.ylabel('Puntuación', fontsize=12)
    
    # Añadir valores en las barras
    for i, (bar, v) in enumerate(zip(bars, valores)):
        plt.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.3f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{reports_path}/{tipo_clasificacion}_metricas_{nombre_modelo}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Matriz de confusión
    plt.figure(figsize=(8, 6))
    cm = resultados['confusion_matrix']
    clases = np.unique(y_test)
    cm_normalized = cm.astype('float') / np.nan_to_num(cm.sum(axis=1, keepdims=True))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=clases)
    
    disp.plot(cmap='Blues', values_format='.2%', ax=plt.gca())
    plt.title(f'Matriz de Confusión - {nombre_modelo} ({tipo_clasificacion.capitalize()})', 
              fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{reports_path}/{tipo_clasificacion}_matriz_confusion_{nombre_modelo}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Distribución de probabilidades
    if tipo_clasificacion == 'binario':
        # Para clasificación binario: distribución de probabilidades para clase positiva
        plt.figure(figsize=(12, 6))
        
        # Separar probabilidades por clase verdadera
        for clase in np.unique(y_test):
            mask = y_test == clase
            if np.any(mask):
                if len(np.unique(y_test)) == 2:
                    # Para SVM binario donde y_prob es 1D
                    if y_prob.ndim == 1:
                        prob_data = y_prob[mask]
                        label = f'Clase real {clase}'
                    else:
                        prob_data = y_prob[mask, 1] if y_prob.shape[1] > 1 else y_prob[mask, 0]
                        label = f'Clase real {clase}'
                else:
                    prob_data = y_prob[mask]
                    label = f'Clase real {clase}'
                
                plt.hist(prob_data, bins=30, alpha=0.7, label=label, density=True)
        
        plt.title(f'Distribución de Probabilidades - {nombre_modelo}')
        plt.xlabel('Probabilidad (Clase Positiva)')
        plt.ylabel('Densidad')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{reports_path}/{tipo_clasificacion}_distribucion_probabilidades_{nombre_modelo}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        
    else:
        # Para clasificación multiclase: distribución por cada clase
        plt.figure(figsize=(15, 10))
        n_clases = y_prob.shape[1]
        
        for i in range(n_clases):
            plt.subplot(1, n_clases, i+1)
            # Separar probabilidades por clase verdadera
            for j in range(n_clases):
                mask = y_test == j
                if np.any(mask):
                    sns.kdeplot(y_prob[mask, i], label=f'Clase real {j}', alpha=0.7)
            
            plt.title(f'Distribución de probabilidad para Clase {i}')
            plt.xlabel('Probabilidad')
            plt.ylabel('Densidad')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{reports_path}/{tipo_clasificacion}_distribucion_probabilidades_{nombre_modelo}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Validación cruzada
    plt.figure(figsize=(10, 6))
    cv_scores = resultados['cv_scores']
    
    plt.boxplot(cv_scores, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', alpha=0.7))
    
    # Determinar la métrica usada en CV
    cv_metric = 'ROC AUC' if tipo_clasificacion == 'binario' else 'Accuracy'
    
    plt.title(f'Distribución de puntuaciones en validación cruzada - {nombre_modelo}')
    plt.ylabel(f'{cv_metric} Score')
    plt.axhline(y=np.mean(cv_scores), color='r', linestyle='--', 
                label=f'Media: {np.mean(cv_scores):.3f}')
    plt.text(1.1, np.mean(cv_scores), f'{np.mean(cv_scores):.3f}', 
            color='r', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{reports_path}/{tipo_clasificacion}_cv_scores_{nombre_modelo}.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Importancia de características (solo para modelos que la soporten)
    if 'feature_importance' in resultados:
        plt.figure(figsize=(12, 8))
        importances = resultados['feature_importance']
        
        # Obtener índices de las características más importantes
        indices = np.argsort(importances)[::-1]
        top_n = min(15, len(importances))  # Mostrar top 15 o todas si hay menos
        
        # Crear el gráfico de barras
        plt.bar(range(top_n), importances[indices[:top_n]])
        plt.title(f'Importancia de Características - {nombre_modelo}')
        plt.xlabel('Características')
        plt.ylabel('Importancia')
        plt.xticks(range(top_n), [f'Feat_{indices[i]}' for i in range(top_n)], rotation=45)
        plt.tight_layout()
        plt.savefig(f'{reports_path}/{tipo_clasificacion}_importancia_caracteristicas_{nombre_modelo}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Curvas ROC y PR (solo para clasificación binario)
    if tipo_clasificacion == 'binario' and 'roc_curve' in resultados:
        plt.figure(figsize=(15, 5))
        
        # Curva ROC
        plt.subplot(1, 2, 1)
        fpr, tpr = resultados['roc_curve']
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {resultados["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Línea aleatoria')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curva ROC')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Curva Precision-Recall
        if 'pr_curve' in resultados:
            plt.subplot(1, 2, 2)
            precision, recall = resultados['pr_curve']
            plt.plot(recall, precision, color='blue', lw=2,
                    label=f'PR curve (AUC = {resultados["pr_auc"]:.3f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Curva Precision-Recall')
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{reports_path}/{tipo_clasificacion}_curvas_roc_pr_{nombre_modelo}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # 7. Resumen del clasificador
    with open(f'{reports_path}/{tipo_clasificacion}_reporte_{nombre_modelo}.txt', 'w', encoding='utf-8') as f:
        f.write(f"===== REPORTE DEL MODELO {nombre_modelo} ({tipo_clasificacion.upper()}) =====\n\n")
        f.write(f"Accuracy: {resultados['accuracy']:.4f}\n")
        f.write(f"Precision: {resultados['precision']:.4f}\n")
        f.write(f"Recall: {resultados['recall']:.4f}\n")
        f.write(f"F1 Score: {resultados['f1']:.4f}\n")
        f.write(f"ROC AUC: {resultados['roc_auc']:.4f}\n")
        
        if tipo_clasificacion == 'binario' and 'pr_auc' in resultados:
            f.write(f"PR AUC: {resultados['pr_auc']:.4f}\n")
        
        f.write("\n===== REPORTE DE CLASIFICACIÓN =====\n\n")
        f.write(resultados['classification_report'])
        f.write("\n\n===== RESULTADOS CV (5-FOLD) =====\n\n")
        f.write(f"Media: {np.mean(resultados['cv_scores']):.4f}\n")
        f.write(f"Desviación estándar: {np.std(resultados['cv_scores']):.4f}\n")
        f.write(f"Min: {np.min(resultados['cv_scores']):.4f}\n")
        f.write(f"Max: {np.max(resultados['cv_scores']):.4f}\n")
        
        # Información adicional para clasificación binario
        if tipo_clasificacion == 'binario':
            cm = resultados['confusion_matrix']
            if cm.shape == (2, 2):  # Asegurar que es matriz 2x2
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                
                f.write(f"\n===== MÉTRICAS ADICIONALES =====\n\n")
                f.write(f"Especificidad: {specificity:.4f}\n")
                f.write(f"Valor Predictivo Negativo: {npv:.4f}\n")
                f.write(f"Verdaderos Negativos: {tn}\n")
                f.write(f"Falsos Positivos: {fp}\n")
                f.write(f"Falsos Negativos: {fn}\n")
                f.write(f"Verdaderos Positivos: {tp}\n")
        
        # Importancia de características
        if 'feature_importance' in resultados:
            f.write("\n\n===== IMPORTANCIA DE CARACTERÍSTICAS (Top 10) =====\n\n")
            importances = resultados['feature_importance']
            indices = np.argsort(importances)[::-1]
            for i in range(min(10, len(importances))):
                f.write(f"{i+1}. Característica {indices[i]}: {importances[indices[i]]:.4f}\n")
    
    print(f"Visualizaciones guardadas en el directorio '{reports_path}/' para {nombre_modelo} ({tipo_clasificacion})")
    return nombre_modelo


def cargar_modelo(ruta_modelo):
    """
    Carga un modelo guardado para hacer predicciones.
    """
    print(f"Cargando modelo desde {ruta_modelo}...")
    return joblib.load(ruta_modelo)

def load_model(models_list: str) -> List:
    """
    Carga los modelos desde el.
    
    Args:
        file_path: 
    """
    try:
        models = []
        for model in models_list:

            if "RandomForest" in model:
                print(f"Intentando cargar el modelo RandomForest: {model}")
                models.append(joblib.load(os.path.join(RF_MODELS_DIR,model)))
            elif "SVM" in model:
                print(f"Intentando cargar el modelo SVM: {model}")
                models.append(joblib.load(os.path.join(SVM_MODELS_DIR,model)))
            elif "GradientBoosting" in model:
                print(f"Intentando cargar el modelo GradientBoosting: {model}")
                models.append(joblib.load(os.path.join(GB_MODELS_DIR,model)))
        # print(models)
        return models
    except:
        print(Exception)
    
