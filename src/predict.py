"""
Módulo para realizar predicciones con el modelo entrenado
"""
import pandas as pd
import numpy as np
import argparse
from utils import cargar_modelo, cargar_datos
from config import *

def predecir_apnea(modelo, datos):
    """
    Realiza predicciones con el modelo guardado.
    """
    # Predicción binaria (apnea obstructiva sí/no)
    prediccion = modelo.predict(datos)
    
    # Probabilidad de apnea obstructiva
    probabilidad = modelo.predict_proba(datos)[:, 1]
    
    return prediccion, probabilidad

def main():

    try:
        modelo = cargar_modelo({MODELS_DIR}/{MODEL_FILENAME})
    except FileNotFoundError:
        print(f"Error: No se encontró el modelo en {args.modelo}")
        print("Por favor, entrene el modelo primero con train.py")
        return
    
    # Preparar datos del paciente
    datos_paciente = {
        # 'nsrr_age': [args.edad],
        'nsrr_bmi': [args.bmi],
        # 'nsrr_bp_systolic': [args.sistolica],
        'nsrr_bp_diastolic': [args.diastolica],
        'nsrr_phrnumar_f1': [args.arousal],
        'nsrr_sex': [1 if args.sexo == 'hombre' else 0],
        'nsrr_race': [args.raza],
        'nsrr_ethnicity': ['not hispanic or latino'],  # Valor por defecto
        'nsrr_current_smoker': [1 if args.fumador_actual == 'si' else 0],
        'nsrr_ever_smoker': [1 if args.fumador_alguna_vez == 'si' else 0]
    }
    
    # Convertir a DataFrame
    df_paciente = pd.DataFrame(datos_paciente)
    
    # Realizar predicción
    prediccion, probabilidad = predecir_apnea(modelo, df_paciente)
    
    # Mostrar resultados
    print("\n===== RESULTADOS DE LA PREDICCIÓN =====")
    print(f"Probabilidad de Apnea Obstructiva del Sueño: {probabilidad[0]:.2%}")
    
    if prediccion[0] == 1:
        print("RESULTADO: POSITIVO - Es probable que el paciente tenga Apnea Obstructiva del Sueño.")
        print("\nRECOMENDACIÓN: Se recomienda realizar una polisomnografía para confirmar el diagnóstico.")
    else:
        print("RESULTADO: NEGATIVO - Es poco probable que el paciente tenga Apnea Obstructiva del Sueño.")
        if probabilidad[0] > 0.3:  # Umbral arbitrario para "zona gris"
            print("\nRECOMENDACIÓN: Aunque el resultado es negativo, hay cierto riesgo. ")
            print("Considere una evaluación adicional si hay síntomas como somnolencia diurna o ronquidos.")
    
    # Mostrar factores de riesgo
    # print("\nFactores de riesgo identificados:")
    # if datos_paciente['nsrr_age'] >= 30:
    #     print("- Obesidad (BMI ≥ 30)")
    # if args.edad > 40:
    #     print("- Edad superior a 40 años")
    # if args.sexo == 'hombre':
    #     print("- Sexo masculino")
    # if args.fumador_actual == 'si':
    #     print("- Fumador actual")
    # if args.sistolica > 140 or args.diastolica > 90:
    #     print("- Hipertensión arterial")

if __name__ == "__main__":
    main()