"""
Módulo para realizar predicciones con el modelo entrenado
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from preprocessing import Preprocessing
from utils import load_model
from config import *

def prepare_patiends(df):
    # Variables categóricas expandidas (incluye nuevas características)
    variables_categoricas = ['bmi_categoria', 'categoria_pa']
    
    # Buscar automáticamente variables categóricas adicionales
    for col in df.columns:
        if (df[col].dtype == 'object' or df[col].dtype == 'category') :
            if col not in variables_categoricas:
                variables_categoricas.append(col)
    
    # Crear copia para evitar modificar el dataframe original
    X = df.copy()

    
    # 4. Manejo mejorado de variables categóricas - Verificar si es necesario
    print("Procesando variables categóricas...")
    for var in variables_categoricas:
        if var in X.columns:
            # Verificar si hay demasiadas categorías únicas
            unique_cats = X[var].nunique()
            if unique_cats > 10:
                print(f"Advertencia: {var} tiene {unique_cats} categorías únicas. Considerando agrupación.")
                # Mantener solo las categorías más frecuentes
                top_cats = X[var].value_counts().head(8).index
                X[var] = X[var].apply(lambda x: x if x in top_cats else 'Otros')
            
            # Crear variables dummy
            try:
                dummies = pd.get_dummies(X[var], prefix=var, drop_first=True, dtype=int)
                X = pd.concat([X, dummies], axis=1)
                X = X.drop(var, axis=1)
            except Exception as e:
                print(f"Error procesando variable categórica {var}: {e}")
                X = X.drop(var, axis=1)
    
    # Eliminar columnas no numéricas restantes con mejor verificación
    non_numeric_cols = []
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            non_numeric_cols.append(col)
        # También verificar si hay strings mezclados en columnas "numéricas"
        elif X[col].dtype == 'object':
            non_numeric_cols.append(col)
    
    if non_numeric_cols:
        print(f"Eliminando columnas no numéricas: {non_numeric_cols}")
        X = X.drop(columns=non_numeric_cols)
    
    # Convertir explícitamente todas las columnas restantes a numéricas
    for col in X.columns:
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        except:
            print(f"No se pudo convertir {col} a numérico, eliminando...")
            X = X.drop(columns=[col])
       
    # 6. NUEVA: Feature selection más sofisticada - Propablemente no sea necesario este bloque
    print("Aplicando selección de características avanzada...")
   
    # Pruebas: Escalado de caracteristicas 
    print("Aplicando RobustScaler...")
    # Guardar los nombres de las columnas antes de la transformación
    original_columns = X.columns
    original_index = X.index

    scaler = RobustScaler()
    X_scaled_array = scaler.fit_transform(X) # Esto devuelve un numpy.ndarray

    # Convertir el numpy.ndarray de vuelta a un DataFrame, manteniendo columnas e índice
    X = pd.DataFrame(X_scaled_array, columns=original_columns, index=original_index)

    return X
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
    # Preparar datos del paciente
    
    """
    nsrr_age,nsrr_sex,nsrr_race,
    nsrr_ethnicity,nsrr_bmi,nsrr_bp_systolic,
    nsrr_bp_diastolic,
    nsrr_current_smoker,nsrr_ever_smoker,
    nsrr_ahi_hp3r_aasm15,apnea_severity
    
    """
    datos_paciente = {
        'nsrr_age': [30],
        'nsrr_sex': ['female'],
        # 'nsrr_race': 'unknown',
        # 'nsrr_ethnicity': 'hispanic or latino',
        'nsrr_bmi': [38.8829556],
        'nsrr_bp_systolic': [97.0],
        'nsrr_bp_diastolic': [66.0],
        'nsrr_current_smoker': ['no'],  
        'nsrr_ever_smoker': ['no'],
    }
    df_paciente = pd.DataFrame(datos_paciente)
    df_paciente = Preprocessing(df_paciente)
    print(f"\n {df_paciente['categoria_pa']}")
    print(f"\n Numero de columnas antes de prepare: {len(df_paciente.columns)} \n {df_paciente.columns} \n")
    
    df_paciente = prepare_patiends(df_paciente)
    
    
    print(f"\n Numero de columnas despues de prepare: {len(df_paciente.columns)} \n {df_paciente.columns} \n")
    print(df_paciente.head())

    models_to_load = ["binario_RandomForest_84_83_83_91.joblib"]

    models = load_model(models_to_load)
    if not models:
        raise Exception("No se encontraron los modelos especificados")
        return
    print("Se cargaron exitosamente los modelos")
    
    
    
    # for i, model in enumerate(models):
        
    
    # # Preparar datos del paciente
    # datos_paciente = {
    #     # 'nsrr_age': [args.edad],
    #     'nsrr_bmi': [args.bmi],
    #     # 'nsrr_bp_systolic': [args.sistolica],
    #     'nsrr_bp_diastolic': [args.diastolica],
    #     'nsrr_phrnumar_f1': [args.arousal],
    #     'nsrr_sex': [1 if args.sexo == 'hombre' else 0],
    #     'nsrr_race': [args.raza],
    #     'nsrr_ethnicity': ['not hispanic or latino'],  # Valor por defecto
    #     'nsrr_current_smoker': [1 if args.fumador_actual == 'si' else 0],
    #     'nsrr_ever_smoker': [1 if args.fumador_alguna_vez == 'si' else 0]
    # }
    
    # # Convertir a DataFrame
    # df_paciente = pd.DataFrame(datos_paciente)
    
    # # Realizar predicción
    # prediccion, probabilidad = predecir_apnea(modelo, df_paciente)
    
    # # Mostrar resultados
    # print("\n===== RESULTADOS DE LA PREDICCIÓN =====")
    # print(f"Probabilidad de Apnea Obstructiva del Sueño: {probabilidad[0]:.2%}")
    
    # if prediccion[0] == 1:
    #     print("RESULTADO: POSITIVO - Es probable que el paciente tenga Apnea Obstructiva del Sueño.")
    #     print("\nRECOMENDACIÓN: Se recomienda realizar una polisomnografía para confirmar el diagnóstico.")
    # else:
    #     print("RESULTADO: NEGATIVO - Es poco probable que el paciente tenga Apnea Obstructiva del Sueño.")
    #     if probabilidad[0] > 0.3:  # Umbral arbitrario para "zona gris"
    #         print("\nRECOMENDACIÓN: Aunque el resultado es negativo, hay cierto riesgo. ")
    #         print("Considere una evaluación adicional si hay síntomas como somnolencia diurna o ronquidos.")


if __name__ == "__main__":
    main()