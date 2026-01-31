import pandas as pd
# from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import  f_classif, mutual_info_classif
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
import numpy as np

from config import TRAIN_TEST_DATA_DIR

def prepare_data(df: pd.DataFrame, apply_pca=False, n_components=None, modo="binario"):
    """
    Prepares data for model training.
    
    Handles data cleaning, categorical encoding, scaling, and class balancing 
    (SMOTE/ADASYN).
    
    Args:
        df (pd.DataFrame): Input dataset.
        apply_pca (bool): PCA flag (currently unused).
        n_components (int/float): PCA components (currently unused).
        modo (str): Target strategy ('multiclase' or 'binario').

    Returns:
        tuple: (X, y) Processed features and target dataframes.
    """
    print("Iniciando preparación de datos...")
    
    # Check for required columns
    required_cols = ['apnea', 'apnea_severity_ordinal']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columnas requeridas faltantes: {missing_cols}.")
    
    # Define target variable
    if modo == 'multiclase':
        y = df['apnea_severity_ordinal'].copy()
        print(f"Modo multiclase - Distribución base:\n{y.value_counts().sort_index()}")
    elif modo == 'binario':
        y = df['apnea'].copy()
        print(f"Modo binario - Distribución base:\n{y.value_counts().sort_index()}")
    else:
        raise ValueError("El modo debe ser 'multiclase' o 'binario'")
    
    # Remove excluded features
    variables_a_excluir = [
        'apnea', 'nsrr_ahi_hp4u_aasm15', 'apnea_severity', 'apnea_severity_ordinal',
        'nsrr_ahi_hp3r_aasm15' 
    ]
    
    # Identify categorical columns
    variables_categoricas = ['bmi_categoria', 'categoria_pa']
    
    for col in df.columns:
        if (df[col].dtype == 'object' or df[col].dtype == 'category') and col not in variables_a_excluir:
            if col not in variables_categoricas:
                variables_categoricas.append(col)
    
    X = df.copy()
    
    # Drop excluded columns
    for var in variables_a_excluir:
        if var in X.columns:
            X = X.drop(var, axis=1)
    
    # Process categorical variables
    print("Procesando variables categóricas...")
    for var in variables_categoricas:
        if var in X.columns:
            # Group rare categories
            unique_cats = X[var].nunique()
            if unique_cats > 10:
                print(f"Alta cardinalidad en {var} ({unique_cats}). Agrupando categorías.")
                top_cats = X[var].value_counts().head(8).index
                X[var] = X[var].apply(lambda x: x if x in top_cats else 'Otros')
            
            # Apply One-Hot Encoding
            try:
                dummies = pd.get_dummies(X[var], prefix=var, drop_first=True, dtype=int)
                X = pd.concat([X, dummies], axis=1)
                X = X.drop(var, axis=1)
            except Exception as e:
                print(f"Fallo en encoding de {var}: {e}")
                X = X.drop(var, axis=1)
    
    # Remove non-numeric columns
    non_numeric_cols = []
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            non_numeric_cols.append(col)
        elif X[col].dtype == 'object':
            non_numeric_cols.append(col)
    
    if non_numeric_cols:
        print(f"Eliminando columnas no numéricas residuales: {non_numeric_cols}")
        X = X.drop(columns=non_numeric_cols)
    
    # Convert to numeric
    for col in X.columns:
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        except:
            print(f"Error convirtiendo columna {col}, eliminando.")
            X = X.drop(columns=[col])
    
    # Apply RobustScaler
    print("Aplicando RobustScaler...")
    original_columns = X.columns
    original_index = X.index

    scaler = RobustScaler()
    X_scaled_array = scaler.fit_transform(X) 

    X = pd.DataFrame(X_scaled_array, columns=original_columns, index=original_index)

    # Balance classes
    print("Ejecutando balanceo de clases...")
    
    class_counts = y.value_counts().sort_index()
    imbalance_ratio = class_counts.max() / class_counts.min()
    print(f"Ratio de desbalance: {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 1.5: 
        if modo == 'multiclase':
            # Use ADASYN for multiclass
            try:
                sampler = ADASYN(random_state=42, n_neighbors=5)
                X_resampled, y_resampled = sampler.fit_resample(X, y)
                
                X = pd.DataFrame(X_resampled, columns=X.columns)
                y = pd.Series(y_resampled)
                
                print("Distribución post-ADASYN:\n", y.value_counts().sort_index())
                
            except Exception as e:
                print(f"Fallo ADASYN ({e}). Aplicando fallback a SMOTE.")
                smote = SMOTE(random_state=42, k_neighbors=5)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                X = pd.DataFrame(X_resampled, columns=X.columns)
                y = pd.Series(y_resampled)
        
        elif modo == 'binario':
            # Use SMOTETomek for binary
            try:
                sampler = SMOTETomek(random_state=42, smote=SMOTE(k_neighbors=5))
                X_resampled, y_resampled = sampler.fit_resample(X, y)
                
                X = pd.DataFrame(X_resampled, columns=X.columns)
                y = pd.Series(y_resampled)
                
                # Export for debugging
                df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
                df_resampled.to_csv("df_resampled.csv")
                
                print("Distribución post-SMOTETomek:\n", y.value_counts().sort_index())
                
            except Exception as e:
                print(f"Fallo SMOTETomek ({e}). Aplicando fallback a SMOTE.")
                smote = SMOTE(random_state=42, k_neighbors=5)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                X = pd.DataFrame(X_resampled, columns=X.columns)
                y = pd.Series(y_resampled)
    
    # Final dataset summary
    print(f"\nResumen del dataset procesado:")
    print(f"Features: {X.shape[1]} | Muestras: {X.shape[0]}")
    print(f"Distribución final:\n{y.value_counts().sort_index()}")
    
    # Ensure all data is numeric
    X = X.select_dtypes(include=[np.number])
    
    # Fill remaining NaNs
    if X.isnull().sum().sum() > 0:
        print("Advertencia: Detectados NaNs, imputando con 0.")
        X = X.fillna(0)
    
    # Handle infinite values
    try:
        if np.isinf(X.values).sum() > 0:
            print("Advertencia: Detectados valores infinitos, reemplazando.")
            X = X.replace([np.inf, -np.inf], 0)
    except TypeError:
        for col in X.columns:
            if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                if np.isinf(X[col]).sum() > 0:
                    X[col] = X[col].replace([np.inf, -np.inf], 0)
    
    # Save processed data
    X.to_csv(f"{TRAIN_TEST_DATA_DIR}/X_{modo}.csv", index=False)
    y.to_csv(f"{TRAIN_TEST_DATA_DIR}/Y_{modo}.csv", index=False)
    
    return X, y