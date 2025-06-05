"""
Módulo mejorado para el preprocesamiento de datos de apnea del sueño
Basado en evidencia científica y mejores prácticas de machine learning
"""
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from config import *


def Preprocessing(df, estrategia_imputation='knn'):
    """
    Realiza el preprocesamiento de datos para predicción de apnea del sueño.
    Mejoras basadas en evidencia científica reciente.
    
    Args:
        df: DataFrame con los datos crudos
        estrategia_imputation: Estrategia para imputar valores faltantes ('knn', 'median', 'mean')
    
    Returns:
        DataFrame procesado
        
    1.- Limpieza y conversion a variables categoricas
    2.- Manejo valores faltantes (mejorado)
    3.- Manejo de valores atipicos (método más conservador)
    4.- Filtrado por edad
    5.- Creacion de caracteristicas medicas adicionales (ampliado)
    6.- Variable objetivo de apnea, binaria (0,1)
    """
    print("Iniciando preprocesamiento de datos para predicción de apnea del sueño...")
    # Hacer una copia para no modificar el original
    df_procesado = df.copy()
    
    # 1. Limpieza y conversión de variables categóricas
    print("Realizando limpieza y transformación de variables categóricas...")
    
    variables_categoricas = ['nsrr_sex', 'nsrr_current_smoker', 'nsrr_ever_smoker']
    
    for var in variables_categoricas:
        if var in df_procesado.columns:
            # Si la variable ya es numérica, continuamos
            if pd.api.types.is_numeric_dtype(df_procesado[var]):
                continue
                
            # Convertir a minúsculas y eliminar espacios
            if df_procesado[var].dtype == 'object':
                df_procesado[var] = df_procesado[var].str.lower() if hasattr(df_procesado[var], 'str') else df_procesado[var]
            
            # Mapeo específico para cada variable
            if var == 'nsrr_sex':
                df_procesado[var] = df_procesado[var].map({'male': 1, 'female': 0})
            elif var == 'nsrr_current_smoker':
                df_procesado[var] = df_procesado[var].map({'yes': 1, 'no': 0, 'not reported': np.nan})
            elif var == 'nsrr_ever_smoker':
                df_procesado[var] = df_procesado[var].map({'yes': 1, 'no': 0, 'not reported': np.nan})
    
    # 2. Manejo de valores faltantes MEJORADO
    print(f"Aplicando estrategia de imputación mejorada: {estrategia_imputation}...")
    
    # Separar variables numéricas para imputación
    variables_numericas = df_procesado.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Eliminar filas con más del 50% de valores faltantes
    threshold_missing = 0.5
    missing_ratio = df_procesado.isnull().sum(axis=1) / len(df_procesado.columns)
    df_procesado = df_procesado[missing_ratio <= threshold_missing]
    print(f"Eliminadas {len(missing_ratio[missing_ratio > threshold_missing])} filas con >50% valores faltantes")
    
    # Imputación para variables numéricas con mejor estrategia
    if estrategia_imputation == 'knn':
        # Usar más vecinos para mayor estabilidad
        imputer = KNNImputer(n_neighbors=7, weights='distance')
        df_procesado[variables_numericas] = imputer.fit_transform(df_procesado[variables_numericas])
    else:
        imputer = SimpleImputer(strategy=estrategia_imputation)
        df_procesado[variables_numericas] = imputer.fit_transform(df_procesado[variables_numericas])
    
    # # Imputación para variables categóricas (moda)
    # for var in variables_categoricas:
    #     if var in df_procesado.columns and df_procesado[var].isnull().sum() > 0:
    #         moda = df_procesado[var].mode()[0]
    #         df_procesado.loc[:, var] = df_procesado[var].fillna(moda)
    
    # 3. Detección y tratamiento de outliers MEJORADO (más conservador)
    print("Detectando y tratando outliers con método más conservador...")
    cantidad_original = len(df_procesado)
    
    # Variables críticas que no deben tener outliers extremos eliminados
    variables_criticas = ['nsrr_ahi_hp3r_aasm15', 'nsrr_bmi', 'nsrr_age']
    
    for var in variables_numericas:
        if var in variables_criticas:
            continue  # No eliminar outliers de variables críticas
            
        # Calcular límites para outliers usando método más conservador 3 IQR
        q1 = df_procesado[var].quantile(0.25)
        q3 = df_procesado[var].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr  # Más conservador
        upper_bound = q3 + 3 * iqr
        
        # Identificar outliers extremos
        mask_outliers = (df_procesado[var] < lower_bound) | (df_procesado[var] > upper_bound)
        outliers_count = mask_outliers.sum()
        
        if outliers_count > 0:
            # Solo eliminar si es menos del 5% de los datos
            if outliers_count < len(df_procesado) * 0.05:
                df_procesado = df_procesado[~mask_outliers]
                print(f"  {var}: {outliers_count} outliers eliminados")
            else:
                print(f"  {var}: {outliers_count} outliers detectados pero no eliminados (>5% datos)")

    print(f"Registros originales: {cantidad_original}, después de tratar outliers: {len(df_procesado)}")

    # 4. Filtrar por criterios de edad (mantenido comentado como original)
    print("Aplicando filtros basados en criterios médicos...")
    if 'nsrr_age' in df_procesado.columns:
        df_procesado = df_procesado[(df_procesado['nsrr_age'] >= EDAD_MIN) & 
                                   (df_procesado['nsrr_age'] <= EDAD_MAX)]
    
    # 5. Creación de características médicas relevantes AMPLIADO
    print("Creando características clínicamente relevantes basadas en evidencia...")
    
    # IMC categorizado según OMS
    if 'nsrr_bmi' in df_procesado.columns:
        df_procesado['bmi_categoria'] = pd.cut(
            df_procesado['nsrr_bmi'], 
            bins=[0, 18.5, 25, 30, 35, 40, float('inf')],
            labels=['Bajo peso', 'Normal', 'Sobrepeso', 'Obesidad I', 'Obesidad II', 'Obesidad III']
        )
        
        # Variable binaria para obesidad (IMC ≥ 30)
        df_procesado['obesidad'] = (df_procesado['nsrr_bmi'] >= 30).astype(int)
        
        # Obesidad severa (IMC ≥ 35) - factor de riesgo mayor
        df_procesado['obesidad_severa'] = (df_procesado['nsrr_bmi'] >= 35).astype(int)
    
    # Categoría de presión arterial (basado en guías de la American Health Association)
    if 'nsrr_bp_systolic' in df_procesado.columns and 'nsrr_bp_diastolic' in df_procesado.columns:
        # Función para categorizar PA
        def categorizar_pa(row):
            sys = row['nsrr_bp_systolic']
            dia = row['nsrr_bp_diastolic']
            
            if sys < 120 and dia < 80:
                return 'Normal'
            elif (sys >= 120 and sys <= 129) and (dia < 80):
                return 'Elevada'
            elif (sys >= 130 and sys <= 139) or (dia >= 80 and dia <=89):
                return 'Hipertension_1'
            elif (sys >= 140 and sys <= 180) or (dia >= 90 and dia <= 120):
                return 'Hipertension_2'
            else:
                return 'Hipertension_Crisis'
         
        df_procesado['categoria_pa'] = df_procesado.apply(categorizar_pa, axis=1)
        
        # Variable binaria para hipertensión
        df_procesado['hipertension'] = df_procesado['categoria_pa'].apply(
            lambda x: 1 if x in ['Hipertension_1', 'Hipertension_2', 'Hipertension_Crisis'] else 0
        )
        
        # NUEVA: Hipertensión severa (mayor predictor)
        df_procesado['hipertension_severa'] = df_procesado['categoria_pa'].apply(
            lambda x: 1 if x in ['Hipertension_2', 'Hipertension_Crisis'] else 0
        )

    # NUEVA: Interacciones importantes basadas en literatura
    if 'nsrr_bmi' in df_procesado.columns and 'nsrr_age' in df_procesado.columns:
        # Edad mayor + obesidad = riesgo muy alto
        df_procesado['edad_obesidad_risk'] = ((df_procesado['nsrr_age'] > 50) & 
                                             (df_procesado['nsrr_bmi'] >= 30)).astype(int)
    
    if 'nsrr_sex' in df_procesado.columns and 'nsrr_age' in df_procesado.columns:
        # Hombres de edad media tienen mayor riesgo
        df_procesado['hombre_edad_media'] = ((df_procesado['nsrr_sex'] == 1) & 
                                           (df_procesado['nsrr_age'] >= 40) & 
                                           (df_procesado['nsrr_age'] <= 65)).astype(int)
    
    # NUEVA: Score de riesgo clínico combinado
    risk_factors = []
    if 'obesidad' in df_procesado.columns:
        risk_factors.append('obesidad')
    if 'hipertension' in df_procesado.columns:
        risk_factors.append('hipertension')
    if 'nsrr_current_smoker' in df_procesado.columns:
        risk_factors.append('nsrr_current_smoker')
    if risk_factors:
        df_procesado['clinical_risk_score'] = df_procesado[risk_factors].sum(axis=1)
    
    
    # 7. Creación de variable objetivo para apnea
    print("Creando variable objetivo para apnea del sueño con umbrales optimizados...")
    
    if 'nsrr_ahi_hp3r_aasm15' in df_procesado.columns:
        # Clasificación de severidad usando criterios AASM actualizados
        df_procesado['apnea_severity'] = pd.cut(
            df_procesado['nsrr_ahi_hp3r_aasm15'], 
            bins=[-np.inf, 5, 15, 30, np.inf],
            labels=['Normal', 'Leve', 'Moderada', 'Severa']
        )
        
        # Variable binaria de apnea (AHI ≥ 5)
        df_procesado['apnea'] = (df_procesado['nsrr_ahi_hp3r_aasm15'] >= 5).astype(int)
        
        # Variable ordinal de severidad (Para prediccion multiclase)
        severity_map = {'Normal': 0, 'Leve': 1, 'Moderada': 2, 'Severa': 3}
        df_procesado['apnea_severity_ordinal'] = df_procesado['apnea_severity'].map(severity_map)


    print(f"Preprocesamiento completado. Registros resultantes: {len(df_procesado)}")
    
    # Mostrar distribución de clases para diagnóstico
    if 'apnea_severity_ordinal' in df_procesado.columns:
        print("\nDistribución de severidad de apnea:")
        print(df_procesado['apnea_severity_ordinal'].value_counts().sort_index())
        
    if 'apnea' in df_procesado.columns:
        print(f"\nDistribución binaria - Sin apnea: {(df_procesado['apnea'] == 0).sum()}, Con apnea: {(df_procesado['apnea'] == 1).sum()}")
    
    df_procesado.to_csv('shhs-harmonized-filtered-preprocessed.csv', index=False)
    return df_procesado

# """
# Módulo mejorado para el preprocesamiento de datos de apnea del sueño
# """
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import  RobustScaler
# from sklearn.impute import KNNImputer, SimpleImputer
# from sklearn.feature_selection import SelectKBest, f_classif, RFE
# from sklearn.ensemble import RandomForestClassifier
# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
# from config import *


# def Preprocessing(df, estrategia_imputation='knn'):
#     """
#     Realiza el preprocesamiento de datos para predicción de apnea del sueño.
    
#     Args:
#         df: DataFrame con los datos crudos
#         estrategia_imputation: Estrategia para imputar valores faltantes ('knn', 'median', 'mean')
    
#     Returns:
#         DataFrame procesado
        
#     1.- Limpieza y conversion a variables categoricas
#     2.- Manejo valores faltantes
#     3.- Manejor de valores atipicos
#     4.- Filtrado por edad
#     5.- Creacion de caracteristicas medicas adicionales
#     6.- Variable objetivo de apnea, binaria (0,1)
#     """
#     print("Iniciando preprocesamiento de datos para predicción de apnea del sueño...")
#     # Hacer una copia para no modificar el original
#     df_procesado = df.copy()
    
#     # 1. Limpieza y conversión de variables categóricas
#     print("Realizando limpieza y transformación de variables categóricas...")
    
#     variables_categoricas = ['nsrr_sex', 'nsrr_current_smoker']
    
#     for var in variables_categoricas:
#         if var in df_procesado.columns:
#             # Si la variable ya es numérica, continuamos
#             if pd.api.types.is_numeric_dtype(df_procesado[var]):
#                 continue
                
#             # Convertir a minúsculas y eliminar espacios
#             if df_procesado[var].dtype == 'object':
#                 df_procesado[var] = df_procesado[var].str.lower() if hasattr(df_procesado[var], 'str') else df_procesado[var]
            
#             # Mapeo específico para cada variable
#             if var == 'nsrr_sex':
#                 df_procesado[var] = df_procesado[var].map({'male': 1, 'female': 0})
#             elif var == 'nsrr_current_smoker':
#                 df_procesado[var] = df_procesado[var].map({'yes': 1, 'no': 0, 'not reported': np.nan})
#             elif var == 'nsrr_ever_smoker':
#                 df_procesado[var] = df_procesado[var].map({'yes': 1, 'no': 0, 'not reported': np.nan})
    
#     # 2. Manejo de valores faltantes
#     print(f"Aplicando estrategia de imputación: {estrategia_imputation}...")
    
#     # Separar variables numéricas para imputación
#     variables_numericas = df_procesado.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
#     # Imputación para variables numéricas
#     if estrategia_imputation == 'knn':
#         imputer = KNNImputer(n_neighbors=5, weights='distance')
#         df_procesado[variables_numericas] = imputer.fit_transform(df_procesado[variables_numericas])
#     else:
#         imputer = SimpleImputer(strategy=estrategia_imputation)
#         df_procesado[variables_numericas] = imputer.fit_transform(df_procesado[variables_numericas])
    
#     # Imputación para variables categóricas (moda)
#     for var in variables_categoricas:
#         if var in df_procesado.columns and df_procesado[var].isnull().sum() > 0:
#             moda = df_procesado[var].mode()[0]
#             df_procesado.loc[:, var] = df_procesado[var].fillna(moda)
    
#     # 3. Detección y tratamiento de outliers
#     print("Detectando y eliminando outliers...")
#     cantidad_original = len(df_procesado)
#     for var in variables_numericas:
#         # Calcular límites para outliers usando el método IQR
#         q1 = df_procesado[var].quantile(0.25)
#         q3 = df_procesado[var].quantile(0.75)
#         iqr = q3 - q1
#         lower_bound = q1 - 3 * iqr
#         upper_bound = q3 + 3 * iqr
        
#         # Identificar outliers extremos
#         mask_outliers = (df_procesado[var] < lower_bound) | (df_procesado[var] > upper_bound)
#         outliers_count = mask_outliers.sum()
        
#         if outliers_count > 0:
#             # Eliminar filas con outliers
#             df_procesado = df_procesado[~mask_outliers]
#             print(f"  {var}: {outliers_count} outliers eliminados")

#     print(f"Registros originales: {cantidad_original}, después de eliminar outliers: {len(df_procesado)}")

#     # 4. Filtrar por criterios de edad (*)
#     # print("Aplicando filtros basados en criterios médicos...")
#     # if 'nsrr_age' in df_procesado.columns:
#     #     df_procesado = df_procesado[(df_procesado['nsrr_age'] >= EDAD_MIN) & 
#     #                                (df_procesado['nsrr_age'] <= EDAD_MAX)]
    
#     # 5. Creación de características médicas relevantes
#     print("Creando características clínicamente relevantes...")
    
#     # IMC categorizado según OMS
#     if 'nsrr_bmi' in df_procesado.columns:
#         df_procesado['bmi_categoria'] = pd.cut(
#             df_procesado['nsrr_bmi'], 
#             bins=[0, 18.5, 25, 30, 35, 40, float('inf')],
#             labels=['Bajo peso', 'Normal', 'Sobrepeso', 'Obesidad I', 'Obesidad II', 'Obesidad III']
#         )
        
#         # Variable binaria para obesidad (IMC ≥ 30)
#         df_procesado['obesidad'] = (df_procesado['nsrr_bmi'] >= 30).astype(int)
    
#     # Categoría de presión arterial (basado en guías de la American Health Association)
#     if 'nsrr_bp_systolic' in df_procesado.columns and 'nsrr_bp_diastolic' in df_procesado.columns:
#         # Función para categorizar PA
#         def categorizar_pa(row):
#             sys = row['nsrr_bp_systolic']
#             dia = row['nsrr_bp_diastolic']
            
#             if sys < 120 and dia < 80:
#                 return 'Normal'
#             elif (sys >= 120 and sys <= 129) or (dia < 80):
#                 return 'Elevada'
#             elif (sys >= 130 and sys <= 139) or (dia >= 80 and dia <=89):
#                 return 'Hipertension_1'
#             elif (sys >= 140 and sys <= 180) or (dia >= 90 and dia <= 120):
#                 return 'Hipertension_2'
#             else:
#                 return 'Hipertension_Crisis'
         
#         df_procesado['categoria_pa'] = df_procesado.apply(categorizar_pa, axis=1)
        
#         # Variable binaria para hipertensión
#         df_procesado['hipertension'] = df_procesado['categoria_pa'].apply(
#             lambda x: 1 if x in ['Hipertension_1', 'Hipertension_2', 'Hipertension_Crisis'] else 0
#         )

    
#     # 7. Creación de variable objetivo para apnea
#     print("Creando variable objetivo para apnea del sueño...")
    
#     if 'nsrr_ahi_hp3r_aasm15' in df_procesado.columns:
#         # Clasificación de severidad usando criterios AASM
#         df_procesado['apnea_severity'] = pd.cut(
#             df_procesado['nsrr_ahi_hp3r_aasm15'], 
#             bins=[-np.inf, 5, 15, 30, np.inf],
#             labels=['Normal', 'Leve', 'Moderada', 'Severa']
#         )
        
#         # Variable binaria de apnea (AHI ≥ 5)
#         df_procesado['apnea'] = (df_procesado['nsrr_ahi_hp3r_aasm15'] >= 5).astype(int)
        
#         # # Variable ordinal de severidad (útil para algunos modelos)
#         severity_map = {'Normal': 0, 'Leve': 1, 'Moderada': 2, 'Severa': 3}
#         df_procesado['apnea_severity_ordinal'] = df_procesado['apnea_severity'].map(severity_map)

#     print(f"Preprocesamiento completado. Registros resultantes: {len(df_procesado)}")
#     df_procesado.to_csv('shhs-harmonized-filtered-preprocessed.csv', index=False)
#     return df_procesado
