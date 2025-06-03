"""
Script simplificado para entrenar modelos de diagnóstico de apnea del sueño
"""
import argparse
from utils import cargar_datos
from preprocessing import preprocesar_datos
from eda import eda_completo
from svm import SVM
from random_forest import RandomForest
from gradient_boost import GradientBoost
from prepare_data import prepare_data
from pso import prepare_data_with_pso_smotetomek
from config import *

def entrenar_modelo(datos_procesados, modelos, modo):
    """Entrena un modelo específico en el modo especificado"""
    
    if "SVM" in modelos:
        model = SVM()
        # X, y = prepare_data(datos_procesados, "SVM", modo=modo)
        # Reemplaza tu función prepare_data() original con:
        pso_params = {
            'n_particles': 20,      # Número de partículas del enjambre
            'max_iterations': 30,   # Iteraciones máximas
            'w': 0.5,              # Peso de inercia
            'c1': 2.0,             # Constante de aceleración personal
            'c2': 2.0,             # Constante de aceleración social
            'min_features': 14     # Mínimo de características a mantener
        }
        X, y = prepare_data_with_pso_smotetomek(
            datos_procesados, 
            algoritmo="SVM", 
            modo=modo,
            use_pso=True,
            pso_params=pso_params
        )
        
        if modo == "binario":
            results = model.train_svm_binary(X, y)
            model.visualizar_resultados_binario(results, nombre_modelo="SVM_binario")
        else:  # multiclase
            results = model.train_svm_multiclase(X, y)
            model.visualizar_resultados_multiclase(results, nombre_modelo='SVM_multiclase')
            
    if "RandomForest" in modelos:
        model = RandomForest()
        X, y = prepare_data(datos_procesados, "Random Forest", modo=modo)
        
        if modo == "binario":
            results = model.train_rf_binary(X, y)
            model.visualizar_resultados_rf(resultados=results, tipo_clasificacion='binaria')
        else:  # multiclase
            results = model.train_rf_multiclase(X, y)
            model.visualizar_resultados_rf(resultados=results, tipo_clasificacion='multiclase')
            
    if "GradientBoost" in modelos:
        model = GradientBoost()
        X, y = prepare_data(datos_procesados, "Gradient Boost", modo="binaria" if modo == "binario" else modo)
        
        if modo == "binario":
            results = model.train_gb_binary(X, y)
            model.visualizar_resultados_gb_binario(results, nombre_modelo="GB Binario")
        else:  # multiclase
            results = model.train_gb_multiclase(X, y)
            model.visualizar_resultados_gb_multiclase(results, nombre_modelo="GB Multiclase")

def main():
    parser = argparse.ArgumentParser(description='Entrenamiento de modelo para apnea del sueño')
    parser.add_argument('--data', type=str, default=DATA_PATH, help=f'Ruta al archivo de datos (default: {DATA_PATH})')
    parser.add_argument('--eda', action='store_true', help='Realizar análisis exploratorio')
    parser.add_argument('--modelo-salida', type=str, default=MODEL_FILENAME, help=f'Nombre del archivo de salida del modelo (default: {MODEL_FILENAME})')
    parser.add_argument('--pre', action='store_true', help="Hacer preprocesamiento")
    parser.add_argument('--train', action='store_true', help="Hacer entrenamiento")
    # parser.add_argument('--modelo', type=str, default="SVM", choices=["SVM", "RandomForest", "GradientBoost"], help="Tipo de modelo a entrenar")
    parser.add_argument('--modo', type=str, default="multiclase", choices=["binario", "multiclase"], help="Modo de clasificación")
    
    """
        Ejemplo: python main.py --pre --train --modelo SVM --modo multiclase
    """
    
    args = parser.parse_args()
    
    # Cargar datos
    # datos = cargar_datos("shhs_hchs_40_20_20_20.csv")
    datos = cargar_datos("shhs_hchs.csv")
    datos = datos.drop(['nsrr_ethnicity', 'nsrr_race'], axis=1)
    print(f"Datos cargados: {len(datos)} pacientes")
    
    # EDA opcional
    if args.eda:
        datos_procesados = eda_completo(datos, "visual_eda")
    
    # Preprocesamiento
    if args.pre:
        datos_procesados = preprocesar_datos(datos)
        if args.eda:
            datos_procesados = eda_completo(datos_procesados, "visual_preprocesado")
        print("Preprocesamiento completado")
    else:
        print("Preprocesamiento omitido")
        return
    
    # Entrenamiento
    if args.train:
        # entrenar_modelo(datos_procesados, args.modelo, args.modo)
        """
            Optiones de modelos a entrenar"SVM", "RandomForest", "GradientBoost"
        """
        modelos_entrenar = ["SVM", "RandomForest"]
        entrenar_modelo(datos_procesados, modelos_entrenar, args.modo)
    else:
        print("Para entrenar el modelo utiliza la opcion --train")

if __name__ == "__main__":
    main()