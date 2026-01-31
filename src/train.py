"""
Main pipeline for model training and evaluation.

Handles the full workflow: Data loading, EDA, Preprocessing, and Training.

Usage:
    python main.py --pre --train --modo multiclase
"""
import time
import argparse

from random_forest import RandomForest
from gradient_boost import GradientBoosting
from svm import SVM
from utils import load_data, create_folders, generate_unique_timestamp, visualizar_resultados_modelo
from preprocessing import Preprocessing
from eda import EDA
from prepare_data import prepare_data
from config import MODELS_TO_TRAIN, VISUAL_EDA_DIR, VISUAL_PREPROCESSED_DIR 

#from pso import prepare_data_with_pso_smotetomek

TIMESTAMP = generate_unique_timestamp()

def train_model(processed_data, models: list, modo: str) -> None:
    """
    Trains selected models based on configuration.

    Prepares features and targets, then executes training and visualization 
    routines for RandomForest, GradientBoosting, or SVM.

    Args:
        processed_data (pd.DataFrame): Cleaned and preprocessed dataset.
        models (list): List of model keys to execute (from config).
        modo (str): Classification strategy ('binario' or 'multiclase').
    """
    X, y = prepare_data(processed_data, modo=modo)
  
    if "RandomForest" in models:
        model = RandomForest()
        if modo == "binario":
            results = model.train_rf_binary(X, y)
            visualizar_resultados_modelo(results, nombre_modelo='RandomForest', tipo_clasificacion=modo)
        else: 
            results = model.train_rf_multiclase(X, y)
            model.visualizar_resultados_rf(resultados=results, tipo_clasificacion='multiclase')
            
    if "GradientBoosting" in models:
        model = GradientBoosting()
        if modo == "binario":
            results = model.train_gb_binary(X, y)
            visualizar_resultados_modelo(results, nombre_modelo='GradientBoosting', tipo_clasificacion=modo)
        else: 
            results = model.train_gb_multiclase(X, y)
            model.visualizar_resultados_gb_multiclase(results, nombre_modelo="GradientBoost_Multiclase")

    if "SVM" in models:
        model = SVM()
        
        if modo == "binario":
            # Hyperparameter optimization (disabled by default)
            # results = model.find_best_parameters_binary(X,y)
            # model.mostrar_resultados_randomized_search_binary(results)  
            results = model.train_svm_binary(X, y)
            visualizar_resultados_modelo(results, nombre_modelo="SVM", tipo_clasificacion=modo)                 
        else: 
            results = model.train_svm_multiclase(X, y)
            model.visualizar_resultados_multiclase(results, nombre_modelo='SVM_multiclase')
            
def main() -> None:
    """
    CLI entry point. Parses arguments, manages directories, and executes the data pipeline.
    """
    parser = argparse.ArgumentParser(description='Pipeline de entrenamiento para apnea del sueño')
    parser.add_argument('--eda', action='store_true', help='Ejecutar análisis exploratorio (EDA)')
    parser.add_argument('--train', action='store_true', help="Ejecutar fase de entrenamiento")
    parser.add_argument('--modo', type=str, default="binario", choices=["binario", "multiclase"], help="Estrategia de clasificación")
    
    start_time = time.time()
    args = parser.parse_args()
    
    # Load data
    data = load_data()
    print(f"Datos cargados: {len(data)} pacientes")
    
    # Setup directories
    create_folders()
    
    # Run Exploratory Data Analysis
    if args.eda:
        EDA(data, VISUAL_EDA_DIR)
    
    # Preprocess data
    processed_data = Preprocessing(data)
    if args.eda:
        EDA(processed_data, VISUAL_PREPROCESSED_DIR)
    print("Preprocesamiento completado")
    
    # Train models
    if args.train:
        # Note: MODELS_TO_TRAIN list is defined in config.py
        train_model(processed_data, MODELS_TO_TRAIN, args.modo)
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Tiempo total de ejecución: {total_time if total_time<60 else total_time/60:.2f}{'s' if total_time<60 else 'm'}")
    else:
        print("Modo inferencia/preprocesamiento. Use --train para entrenar modelos.")

if __name__ == "__main__":
    main()