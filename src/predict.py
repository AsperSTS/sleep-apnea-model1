"""
Inference module for apnea prediction models.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from preprocessing import Preprocessing
from utils import load_model
from config import *

def prepare_patiends(df):
    """
    Runs the data transformation pipeline for inference.
    
    Handles categorical encoding, cleans non-numeric data, and applies 
    robust scaling to normalize features.

    Args:
        df (pd.DataFrame): Raw patient data.

    Returns:
        pd.DataFrame: Transformed and normalized data.
    """
    # Define base categorical variables
    variables_categoricas = ['bmi_categoria', 'categoria_pa']
    
    # Auto-detect object or category columns
    for col in df.columns:
        if (df[col].dtype == 'object' or df[col].dtype == 'category'):
            if col not in variables_categoricas:
                variables_categoricas.append(col)
    
    # Create working copy
    X = df.copy()

    # Process categorical variables
    print("Procesando variables categóricas...")
    for var in variables_categoricas:
        if var in X.columns:
            # Check cardinality
            unique_cats = X[var].nunique()
            if unique_cats > 10:
                print(f"Advertencia: {var} con alta cardinalidad ({unique_cats}). Aplicando agrupación.")
                # Keep top 8 categories, group others
                top_cats = X[var].value_counts().head(8).index
                X[var] = X[var].apply(lambda x: x if x in top_cats else 'Otros')
            
            # Apply One-Hot Encoding
            try:
                dummies = pd.get_dummies(X[var], prefix=var, drop_first=True, dtype=int)
                X = pd.concat([X, dummies], axis=1)
                X = X.drop(var, axis=1)
            except Exception as e:
                print(f"Error en encoding de {var}: {e}")
                X = X.drop(var, axis=1)
    
    # Identify non-numeric columns
    non_numeric_cols = []
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            non_numeric_cols.append(col)
        elif X[col].dtype == 'object':
            non_numeric_cols.append(col)
    
    if non_numeric_cols:
        print(f"Depurando columnas no numéricas: {non_numeric_cols}")
        X = X.drop(columns=non_numeric_cols)
    
    # Convert to numeric types
    for col in X.columns:
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        except:
            print(f"Fallo conversión numérica en {col}, eliminando.")
            X = X.drop(columns=[col])
       
    # Normalize data
    print("Aplicando RobustScaler...")
    original_columns = X.columns
    original_index = X.index

    scaler = RobustScaler()
    X_scaled_array = scaler.fit_transform(X) 

    # Rebuild DataFrame with original structure
    X = pd.DataFrame(X_scaled_array, columns=original_columns, index=original_index)

    return X

def predecir_apnea(modelo, datos):
    """
    Runs inference on processed data.

    Args:
        modelo: Trained model object (e.g., RandomForest, GBM).
        datos (pd.DataFrame): Processed features.

    Returns:
        tuple: (class_prediction, positive_class_probability)
    """
    # Predict class (0/1)
    prediccion = modelo.predict(datos)
    
    # Get probability
    probabilidad = modelo.predict_proba(datos)[:, 1]
    
    return prediccion, probabilidad

def main():
    """
    Main execution flow.
    
    Generates mock data, runs preprocessing, and loads models.
    """
    # Define mock patient data
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
    
    # Run preprocessing pipeline
    df_paciente = Preprocessing(df_paciente)
    print(f"\n {df_paciente['categoria_pa']}")
    print(f"\n Columnas pre-transformación: {len(df_paciente.columns)} \n {df_paciente.columns} \n")
    
    df_paciente = prepare_patiends(df_paciente)
    
    print(f"\n Columnas post-transformación: {len(df_paciente.columns)} \n {df_paciente.columns} \n")
    print(df_paciente.head())

    # Load artifacts
    models_to_load = ["binario_RandomForest_84_83_83_91.joblib"]
    models = load_model(models_to_load)
    
    if not models:
        raise Exception("Fallo en carga de modelos: Lista vacía o archivos no encontrados.")
        return
        
    print("Modelos cargados correctamente.")
    
    # Inference loop (Pending production implementation)
    # for i, model in enumerate(models):
    #     prediccion, probabilidad = predecir_apnea(model, df_paciente)
    #     ...

if __name__ == "__main__":
    main()