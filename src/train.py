"""
Script simplificado para entrenar models de diagnóstico de apnea del sueño
"""
import time
import argparse

from random_forest import RandomForest
from gradient_boost import GradientBoosting
from svm import SVM
from utils import load_data, create_folders, generate_unique_timestamp,visualizar_resultados_modelo
from preprocessing import Preprocessing
from eda import EDA
from prepare_data import prepare_data
from config import MODELS_TO_TRAIN, VISUAL_EDA_DIR, VISUAL_PREPROCESSED_DIR 

#from pso import prepare_data_with_pso_smotetomek

TIMESTAMP = generate_unique_timestamp()

def train_model(processed_data, models: list, modo: str) -> None:
    """Entrena un modelo específico en el modo especificado"""

    X, y = prepare_data(processed_data, modo=modo)

    
    if "SVM" in models:
        model = SVM()
        
        if modo == "binario":
            # results = model.find_best_parameters_binary(X,y)
            # model.mostrar_resultados_randomized_search_binary(results)  
            results = model.train_svm_binary(X, y)
            visualizar_resultados_modelo(results, nombre_modelo="SVM",tipo_clasificacion=modo)                 
        else:  # multiclase
            results = model.train_svm_multiclase(X, y)
            model.visualizar_resultados_multiclase(results, nombre_modelo='SVM_multiclase')
            
    if "RandomForest" in models:
        model = RandomForest()
        if modo == "binario":
            results = model.train_rf_binary(X, y)
            visualizar_resultados_modelo(results,nombre_modelo='RandomForest', tipo_clasificacion=modo)
        else:  # multiclase
            results = model.train_rf_multiclase(X, y)
            model.visualizar_resultados_rf(resultados=results, tipo_clasificacion='multiclase')
            
    if "GradientBoosting" in models:
        model = GradientBoosting()
        if modo == "binario":
            results = model.train_gb_binary(X, y)
            visualizar_resultados_modelo(results,nombre_modelo='GradientBoosting', tipo_clasificacion=modo)
        else:  # multiclase
            results = model.train_gb_multiclase(X, y)
            model.visualizar_resultados_gb_multiclase(results, nombre_modelo="GradientBoost_Multiclase")

def main() -> None:
    parser = argparse.ArgumentParser(description='Entrenamiento de modelo para apnea del sueño')
    parser.add_argument('--eda', action='store_true', help='Realizar análisis exploratorio')
    parser.add_argument('--train', action='store_true', help="Hacer entrenamiento")
    parser.add_argument('--modo', type=str, default="multiclase", choices=["binario", "multiclase"], help="Modo de clasificación")
    
    """
        Ejemplo: python main.py --pre --train --modelo SVM --modo multiclase
    """
    start_time = time.time()
    args = parser.parse_args()
    
    # Cargar data
    data = load_data()
    print(f"Datos cargados: {len(data)} pacientes")
    
    # Se crean todas las carpetas necesarias para el proyecto
    create_folders()
    
    # EDA opcional
    if args.eda:
        EDA(data, VISUAL_EDA_DIR)
    
    # Preprocesamiento
    processed_data = Preprocessing(data)
    if args.eda:
        EDA(processed_data, VISUAL_PREPROCESSED_DIR)
    print("Preprocesamiento completado")
    
    # Entrenamiento
    if args.train:
        """
            Optiones de models a entrenar"SVM", "RandomForest", "GradientBoosting"
        """
        train_model(processed_data, MODELS_TO_TRAIN, args.modo)
        
        end_time = time.time()
        total_time = end_time-start_time
        print(f"Tiempo de entrenamiento: {total_time if total_time<60 else total_time/60}{'s' if total_time<60 else 'm'}")
    else:
        print("Para entrenar el modelo utiliza la opcion --train")

if __name__ == "__main__":
    main()