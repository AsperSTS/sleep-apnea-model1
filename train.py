"""
Script principal para entrenar el modelo de diagnóstico de apnea del sueño
"""
import argparse
from utils import cargar_datos, guardar_modelo, visualizar_resultados, guardar_lista_csv
from preprocessing import preprocesar_datos, preparar_datos_modelo
from eda import eda_completo
from modeling import construir_modelo
from config import *


def main():
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Entrenamiento de modelo para apnea del sueño')
    parser.add_argument('--data', type=str, default=DATA_PATH, 
                        help=f'Ruta al archivo de datos (default: {DATA_PATH})')
    parser.add_argument('--solo-obstructiva', action='store_true', 
                        help='Entrenar solo para apnea obstructiva, excluyendo casos centrales')
    parser.add_argument('--sin-analisis', action='store_true',
                        help='Omitir el análisis exploratorio para ahorrar tiempo')
    parser.add_argument('--modelo-salida', type=str, default=MODEL_FILENAME,
                        help=f'Nombre del archivo de salida del modelo (default: {MODEL_FILENAME})')
    parser.add_argument('--hacer-entrenamiento', default=0, type=int,help="1 para hacer entrenamiento, 0 para evitarlo")
    # Analizar argumentos
    args = parser.parse_args()
    
    # Cargar datos
    datos = cargar_datos(args.data)
    
    # Preprocesar datos
    datos_procesados = preprocesar_datos(datos)
    
    # Análisis exploratorio (opcional)
    if not args.sin_analisis:
        # datos_procesados = analisis_exploratorio(datos_procesados)
        datos_procesados = eda_completo(datos_procesados)
    
    
    hacer_entrenamiento = args.hacer_entrenamiento
    if hacer_entrenamiento:
        # Preparar datos para modelo
        X_train, X_test, y_train, y_test, selected_features = preparar_datos_modelo(
            datos_procesados, solo_obstructiva=args.solo_obstructiva
        )
        
        guardar_lista_csv("selected_features","selected_features.csv", selected_features)
        print(selected_features)
        
        # Guardar selected_features para poder usarlo despues en el modelo
        # Construir y entrenar modelo
        # resultados, mejor_modelo_nombre = construir_modelo(X_train, y_train, X_test, y_test)
        # Construir y entrenar modelo
        resultados, mejor_modelo_nombre = construir_modelo(X_train, y_train, X_test, y_test, selected_features)
        # Visualizar resultados
        mejor_modelo_nombre = visualizar_resultados(resultados, X_test, y_test)
        
        # Guardar el mejor modelo
        mejor_modelo = resultados[mejor_modelo_nombre]['modelo']
        ruta_modelo = guardar_modelo(mejor_modelo, args.modelo_salida)
        
        print(f"\n¡Entrenamiento completado con éxito!")
        print(f"El mejor modelo ({mejor_modelo_nombre}) ha sido guardado en: {ruta_modelo}")
        print("\nPara realizar predicciones, utilice el script predict.py")
        print("Ejemplo:")
        print("python predict.py --edad 45 --sexo hombre --bmi 28.5 --sistolica 135 --diastolica 85 --arousal 12.3 --raza blanco --fumador_actual no --fumador_alguna_vez si")

if __name__ == "__main__":
    main()