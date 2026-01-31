"""
Preprocessing pipeline for the apnea dataset.
Handles cleaning, imputation, and feature engineering.
"""
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from config import *

def Preprocessing(df, estrategia_imputation='knn'):
    """
    Executes the data transformation pipeline.
    
    Standardizes categories, handles missing values, removes outliers, 
    and generates clinical features.

    Args:
        df (pd.DataFrame): Raw dataset.
        estrategia_imputation (str): Imputation method ('knn', 'mean', 'median').

    Returns:
        pd.DataFrame: Processed dataset ready for modeling.
    """
    print("Iniciando pipeline de preprocesamiento...")
    
    df_procesado = df.copy()
    
    # 1. Standardize categorical variables
    print("Procesando variables categóricas...")
    
    variables_categoricas = ['nsrr_sex', 'nsrr_current_smoker', 'nsrr_ever_smoker', 'nsrr_race', 'nsrr_ethnicity']
    
    for var in variables_categoricas:
        if var in df_procesado.columns:
            if pd.api.types.is_numeric_dtype(df_procesado[var]):
                continue
                
            # Normalize strings
            if df_procesado[var].dtype == 'object':
                df_procesado[var] = df_procesado[var].str.lower() if hasattr(df_procesado[var], 'str') else df_procesado[var]
            
            # Map values to numeric codes
            if var == 'nsrr_sex':
                df_procesado[var] = df_procesado[var].map({'male': 1, 'female': 0})
            elif var == 'nsrr_current_smoker':
                df_procesado[var] = df_procesado[var].map({'yes': 1, 'no': 0, 'not reported': np.nan})
            elif var == 'nsrr_ever_smoker':
                df_procesado[var] = df_procesado[var].map({'yes': 1, 'no': 0, 'not reported': np.nan})
            
            elif var == 'nsrr_race':
                # Filter undefined racial categories
                df_procesado = df_procesado[~df_procesado["nsrr_race"].isin(["unknown", "other"])]
                df_procesado[var] = df_procesado[var].map({
                    'white': 0, 
                    'black or african american': 1,
                    'american indian or alaska native': 2, 
                    'multiple': 3,
                    'native hawaiian or other pacific islander': 4, 
                    'asian': 5
                })
            elif var == 'nsrr_ethnicity':
                df_procesado[var] = df_procesado[var].map({'not hispanic or latino': 0, 'hispanic or latino': 1})   
    
    # 2. Handle missing values
    print(f"Ejecutando imputación ({estrategia_imputation})...")
    
    variables_numericas = df_procesado.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Drop rows with >50% missing data
    threshold_missing = 0.5
    missing_ratio = df_procesado.isnull().sum(axis=1) / len(df_procesado.columns)
    df_procesado = df_procesado[missing_ratio <= threshold_missing]
    print(f"Registros eliminados por falta de datos: {len(missing_ratio[missing_ratio > threshold_missing])}")
    
    # Apply imputation
    if estrategia_imputation == 'knn':
        imputer = KNNImputer(n_neighbors=7, weights='distance')
        df_procesado[variables_numericas] = imputer.fit_transform(df_procesado[variables_numericas])
    else:
        imputer = SimpleImputer(strategy=estrategia_imputation)
        df_procesado[variables_numericas] = imputer.fit_transform(df_procesado[variables_numericas])

    # 3. Filter outliers
    print("Filtrando outliers (IQR extendido)...")
    cantidad_original = len(df_procesado)
    
    # Skip critical variables
    variables_criticas = ['nsrr_ahi_hp3r_aasm15', 'nsrr_bmi', 'nsrr_age']
    
    for var in variables_numericas:
        if var in variables_criticas:
            continue 
            
        q1 = df_procesado[var].quantile(0.25)
        q3 = df_procesado[var].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr 
        upper_bound = q3 + 3 * iqr
        
        mask_outliers = (df_procesado[var] < lower_bound) | (df_procesado[var] > upper_bound)
        outliers_count = mask_outliers.sum()
        
        if outliers_count > 0:
            # Only remove outliers if they represent <5% of the data
            if outliers_count < len(df_procesado) * 0.05:
                df_procesado = df_procesado[~mask_outliers]
                print(f"  {var}: {outliers_count} eliminados.")
            else:
                print(f"  {var}: {outliers_count} detectados (conservados por volumen).")

    print(f"Reducción de muestras por outliers: {cantidad_original} -> {len(df_procesado)}")

    # 4. Filter by age range
    if 'nsrr_age' in df_procesado.columns:
        df_procesado = df_procesado[(df_procesado['nsrr_age'] >= EDAD_MIN) & 
                                    (df_procesado['nsrr_age'] <= EDAD_MAX)]
    
    # 5. Generate derived features
    print("Generando características derivadas...")
    
    # Categorize BMI (WHO standards)
    if 'nsrr_bmi' in df_procesado.columns:
        df_procesado['bmi_categoria'] = pd.cut(
            df_procesado['nsrr_bmi'], 
            bins=[0, 18.5, 25, 30, 35, 40, float('inf')],
            labels=['Bajo peso', 'Normal', 'Sobrepeso', 'Obesidad I', 'Obesidad II', 'Obesidad III']
        )
        
        df_procesado['obesidad'] = (df_procesado['nsrr_bmi'] >= 30).astype(int)
        df_procesado['obesidad_severa'] = (df_procesado['nsrr_bmi'] >= 35).astype(int)
    
    # Categorize Blood Pressure (AHA guidelines)
    if 'nsrr_bp_systolic' in df_procesado.columns and 'nsrr_bp_diastolic' in df_procesado.columns:
        def categorizar_pa(row):
            sys = row['nsrr_bp_systolic']
            dia = row['nsrr_bp_diastolic']
            
            if sys < 120 and dia < 80: return 'Normal'
            elif (sys >= 120 and sys <= 129) and (dia < 80): return 'Elevada'
            elif (sys >= 130 and sys <= 139) or (dia >= 80 and dia <=89): return 'Hipertension_1'
            elif (sys >= 140 and sys <= 180) or (dia >= 90 and dia <= 120): return 'Hipertension_2'
            else: return 'Hipertension_Crisis'
         
        df_procesado['categoria_pa'] = df_procesado.apply(categorizar_pa, axis=1)
        
        df_procesado['hipertension'] = df_procesado['categoria_pa'].apply(
            lambda x: 1 if x in ['Hipertension_1', 'Hipertension_2', 'Hipertension_Crisis'] else 0
        )
        
        df_procesado['hipertension_severa'] = df_procesado['categoria_pa'].apply(
            lambda x: 1 if x in ['Hipertension_2', 'Hipertension_Crisis'] else 0
        )

    # Create risk interaction flags
    if 'nsrr_bmi' in df_procesado.columns and 'nsrr_age' in df_procesado.columns:
        df_procesado['edad_obesidad_risk'] = ((df_procesado['nsrr_age'] > 50) & 
                                              (df_procesado['nsrr_bmi'] >= 30)).astype(int)
    
    if 'nsrr_sex' in df_procesado.columns and 'nsrr_age' in df_procesado.columns:
        df_procesado['hombre_edad_media'] = ((df_procesado['nsrr_sex'] == 1) & 
                                             (df_procesado['nsrr_age'] >= 40) & 
                                             (df_procesado['nsrr_age'] <= 65)).astype(int)
    
    # Calculate cumulative clinical risk score
    risk_factors = []
    if 'obesidad' in df_procesado.columns: risk_factors.append('obesidad')
    if 'hipertension' in df_procesado.columns: risk_factors.append('hipertension')
    if 'nsrr_current_smoker' in df_procesado.columns: risk_factors.append('nsrr_current_smoker')
    
    if risk_factors:
        df_procesado['clinical_risk_score'] = df_procesado[risk_factors].sum(axis=1)
    
   
    # 6. Define target variables (Apnea)
    print("Calculando variable objetivo (AHI)...")
    
    if 'nsrr_ahi_hp3r_aasm15' in df_procesado.columns:
        # Categorize severity based on AHI thresholds
        df_procesado['apnea_severity'] = pd.cut(
            df_procesado['nsrr_ahi_hp3r_aasm15'], 
            bins=[-np.inf, 5, 15, 30, np.inf],
            labels=['Normal', 'Leve', 'Moderada', 'Severa']
        )
        
        # Binary target (AHI >= 5)
        df_procesado['apnea'] = (df_procesado['nsrr_ahi_hp3r_aasm15'] >= 5).astype(int)
        
        # Ordinal target
        severity_map = {'Normal': 0, 'Leve': 1, 'Moderada': 2, 'Severa': 3}
        df_procesado['apnea_severity_ordinal'] = df_procesado['apnea_severity'].map(severity_map)

    # Print summary statistics and save result
    print(f"Preprocesamiento finalizado. Total registros: {len(df_procesado)}")
    
    if 'apnea_severity_ordinal' in df_procesado.columns:
        print(f"Distribución ordinal:\n{df_procesado['apnea_severity_ordinal'].value_counts().sort_index()}")
        
    if 'apnea' in df_procesado.columns:
        print(f"Distribución binaria: Positivos={(df_procesado['apnea'] == 1).sum()}, Negativos={(df_procesado['apnea'] == 0).sum()}")
    
    df_procesado.to_csv('shhs-harmonized-filtered-preprocessed.csv', index=False)
    
    return df_procesado