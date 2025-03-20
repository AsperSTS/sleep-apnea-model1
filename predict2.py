"""
Módulo para realizar predicciones de apnea obstructiva del sueño
"""
import pandas as pd
import numpy as np
from utils import cargar_modelo
from config import MODELS_DIR, MODEL_FILENAME

# Ruta predeterminada al modelo
RUTA_MODELO = f"{MODELS_DIR}/{MODEL_FILENAME}"

def cargar_modelo_apnea(ruta_modelo=RUTA_MODELO):
    """Carga el modelo entrenado de apnea"""
    return cargar_modelo(ruta_modelo)

def predecir_paciente_simple(datos_dict, modelo=None):
    """
    Predice apnea para un paciente usando un diccionario simple de datos.
    
    Args:
        datos_dict: Diccionario con los datos del paciente (ej: {'edad': 45, 'bmi': 30})
        modelo: Modelo cargado (opcional, se cargará si no se proporciona)
    
    Returns:
        prediccion: 0 (negativo) o 1 (positivo)
        probabilidad: Probabilidad de apnea (0-1)
    """
    # Cargar modelo si no se proporcionó
    if modelo is None:
        modelo = cargar_modelo_apnea()
    
    # Mapear nombres comunes a nombres de columnas del modelo
    mapeo_nombres = {
        'edad': 'nsrr_age',
        'sexo': 'nsrr_sex',
        'bmi': 'nsrr_bmi',
        'sistolica': 'nsrr_bp_systolic',
        'diastolica': 'nsrr_bp_diastolic',
        'arousal': 'nsrr_phrnumar_f1',
        'raza': 'nsrr_race',
        'fumador_actual': 'nsrr_current_smoker',
        'fumador_alguna_vez': 'nsrr_ever_smoker'
    }
    
    # Convertir datos a formato del modelo
    datos_modelo = {}
    
    # Procesar cada clave en el diccionario
    for clave, valor in datos_dict.items():
        # Si es un nombre común, usar el mapeo
        if clave in mapeo_nombres:
            nombre_columna = mapeo_nombres[clave]
            
            # Convertir valores de texto a numéricos si es necesario
            if clave == 'sexo':
                valor_convertido = 1 if valor.lower() in ['hombre', 'masculino', 'h', 'm'] else 0
            elif clave in ['fumador_actual', 'fumador_alguna_vez']:
                valor_convertido = 1 if valor.lower() in ['si', 's', 'yes', 'y', 'true', '1'] else 0
            else:
                valor_convertido = valor
            
            datos_modelo[nombre_columna] = [valor_convertido]
        else:
            # Usar el nombre directamente si no está en el mapeo
            datos_modelo[clave] = [valor]
    
    # Añadir ethnicity por defecto si no está presente
    if 'nsrr_ethnicity' not in datos_modelo:
        datos_modelo['nsrr_ethnicity'] = ['not hispanic or latino']
    
    # Convertir a DataFrame
    df_paciente = pd.DataFrame(datos_modelo)
    
    # Realizar predicción
    prediccion = modelo.predict(df_paciente)[0]
    probabilidad = modelo.predict_proba(df_paciente)[0, 1]
    
    return prediccion, probabilidad

def predecir_desde_csv_simple(ruta_csv, ruta_modelo=RUTA_MODELO):
    """
    Predice apnea para múltiples pacientes desde un CSV.
    
    Args:
        ruta_csv: Ruta al archivo CSV con datos de pacientes
        ruta_modelo: Ruta al modelo entrenado
    
    Returns:
        DataFrame con los datos originales y las predicciones
    """
    # Cargar modelo
    modelo = cargar_modelo_apnea(ruta_modelo)
    
    # Cargar datos
    datos = pd.read_csv(ruta_csv)
    
    # Realizar predicciones
    predicciones = modelo.predict(datos)
    probabilidades = modelo.predict_proba(datos)[:, 1]
    
    # Añadir resultados
    datos['prediccion'] = predicciones
    datos['probabilidad'] = probabilidades
    
    # Guardar resultados
    ruta_resultados = ruta_csv.replace('.csv', '_resultados.csv')
    datos.to_csv(ruta_resultados, index=False)
    
    return datos

def imprimir_resultados(prediccion, probabilidad, datos=None):
    """
    Imprime los resultados de manera amigable
    """
    print("\n===== RESULTADOS DE LA PREDICCIÓN =====")
    print(f"Probabilidad de Apnea Obstructiva del Sueño: {probabilidad:.2%}")
    
    if prediccion == 1:
        print("RESULTADO: POSITIVO - Es probable que el paciente tenga Apnea Obstructiva del Sueño.")
        print("RECOMENDACIÓN: Realizar polisomnografía para confirmar el diagnóstico.")
    else:
        print("RESULTADO: NEGATIVO - Es poco probable que el paciente tenga Apnea Obstructiva del Sueño.")
        if probabilidad > 0.3:
            print("RECOMENDACIÓN: Considere evaluación adicional si hay síntomas como somnolencia o ronquidos.")
    
    # Mostrar factores de riesgo si hay datos
    if datos:
        print("\nFactores de riesgo identificados:")
        if datos.get('bmi', 0) >= 30 or datos.get('nsrr_bmi', 0) >= 30:
            print("- Obesidad (BMI ≥ 30)")
        if datos.get('edad', 0) > 40 or datos.get('nsrr_age', 0) > 40:
            print("- Edad superior a 40 años")
        if datos.get('sexo', '') == 'hombre' or datos.get('nsrr_sex', 0) == 1:
            print("- Sexo masculino")
        if datos.get('fumador_actual', '') in ['si', 's', 'yes', 'y', '1', 1] or datos.get('nsrr_current_smoker', 0) == 1:
            print("- Fumador actual")
        if (datos.get('sistolica', 0) > 140 or datos.get('nsrr_bp_systolic', 0) > 140 or
            datos.get('diastolica', 0) > 90 or datos.get('nsrr_bp_diastolic', 0) > 90):
            print("- Hipertensión arterial")

# Ejemplos de uso
if __name__ == "__main__":
    # Ejemplo simple: un paciente
    print("Ejemplo 1: Un solo paciente")
    datos_paciente = {
        'edad': 30,
        'sexo': 'hombre',
        'bmi': 32,
        'sistolica': 135,
        'diastolica': 85,
        'arousal': 15,
        'fumador_actual': 'si'
    }
    
    prediccion, probabilidad = predecir_paciente_simple(datos_paciente)
    imprimir_resultados(prediccion, probabilidad, datos_paciente)
    
    # Ejemplo 2: múltiples pacientes desde CSV
    print("\nEjemplo 2: Múltiples pacientes desde CSV")
    print("Para usar, descomenta las siguientes líneas:")
    print("# resultados = predecir_desde_csv_simple('pacientes.csv')")
    print("# print(f'Resultados guardados en pacientes_resultados.csv')")