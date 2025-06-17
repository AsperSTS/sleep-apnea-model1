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
    Prepara los datos para el entrenamiento con mejoras basadas en evidencia científica.
    
    Mejoras implementadas:
    - Mejor manejo de desbalance de clases con técnicas híbridas
    - Feature selection más sofisticada
    - Transformaciones para mejorar distribuciones
    - Validación más robusta de datos
    
    Args:
        df: DataFrame con datos preprocesados
        algoritmo: Nombre del algoritmo para logging
        apply_pca: Si True, aplica PCA para reducción de dimensionalidad
        n_components: Componentes PCA o varianza explicada si es float
        modo: 'multiclase' o 'binario'
    Returns:
        X, y: Conjuntos de datos preparados para entrenamiento
    """
    print("Preparando datos para modelado de apnea del sueño con mejoras...")
    
    # 1. Validación inicial mejorada
    required_cols = ['apnea', 'apnea_severity_ordinal']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Variables faltantes: {missing_cols}. Ejecute Preprocessing primero")
    
    # 2. Definir variable objetivo según modo
    if modo == 'multiclase':
        y = df['apnea_severity_ordinal'].copy()
        print(f"Modo multiclase - Distribución original:")
        print(y.value_counts().sort_index())
    elif modo == 'binario':
        y = df['apnea'].copy()
        print(f"Modo binario - Distribución original:")
        print(y.value_counts().sort_index())
    else:
        raise ValueError("Modo debe ser 'multiclase' o 'binario'")
    
    # 3. Preparar variables predictoras con lista ampliada de exclusiones
    variables_a_excluir = [
        'apnea', 'nsrr_ahi_hp4u_aasm15', 'apnea_severity', 'apnea_severity_ordinal',
        'nsrr_ahi_hp3r_aasm15' #, 'apnea_significativa', 'apnea_severa'  # Nuevas variables objetivo
    ]
    
    # Variables categóricas expandidas (incluye nuevas características)
    variables_categoricas = ['bmi_categoria', 'categoria_pa']
    
    # Buscar automáticamente variables categóricas adicionales
    for col in df.columns:
        if (df[col].dtype == 'object' or df[col].dtype == 'category') and col not in variables_a_excluir:
            if col not in variables_categoricas:
                variables_categoricas.append(col)
    
    # Crear copia para evitar modificar el dataframe original
    X = df.copy()
    
    # Eliminar variables objetivo y relacionadas
    for var in variables_a_excluir:
        if var in X.columns:
            X = X.drop(var, axis=1)
    
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
    
    # 5. NUEVA: Transformación de datos para mejorar distribuciones
    # print("Aplicando transformaciones para mejorar distribuciones...")
    # # Identificar variables con distribuciones muy sesgadas
    # skewed_features = []
    # for col in X.select_dtypes(include=[np.number]).columns:
    #     if abs(X[col].skew()) > 2:  # Sesgo significativo
    #         skewed_features.append(col)
    
    # if skewed_features:
    #     print(f"Aplicando transformación Yeo-Johnson a {len(skewed_features)} variables sesgadas")
    #     pt = PowerTransformer(method='yeo-johnson', standardize=False)
    #     X[skewed_features] = pt.fit_transform(X[skewed_features])
    
    # 6. NUEVA: Feature selection más sofisticada - Propablemente no sea necesario este bloque
    print("Aplicando selección de características avanzada...")
    
    # # Solo aplicar si tenemos suficientes características
    # if X.shape[1] > 50:
    #     # Combinación de métodos de selección
    #     # 1. Selección basada en información mutua
    #     mi_selector = SelectKBest(score_func=mutual_info_classif, k=min(50, X.shape[1]//2))
    #     X_mi = mi_selector.fit_transform(X, y)
    #     selected_features_mi = X.columns[mi_selector.get_support()].tolist()
        
    #     # 2. Selección basada en F-score
    #     f_selector = SelectKBest(score_func=f_classif, k=min(50, X.shape[1]//2))
    #     X_f = f_selector.fit_transform(X, y)
    #     selected_features_f = X.columns[f_selector.get_support()].tolist()
        
    #     # Combinar características seleccionadas
    #     combined_features = list(set(selected_features_mi + selected_features_f))
    #     X = X[combined_features]
    #     print(f"Características reducidas de {df.shape[1]} a {len(combined_features)} usando selección híbrida")
    
    # # 7. Aplicar PCA si se solicita (mejorado)
    # if apply_pca:
    #     print(f"Aplicando PCA con componentes={n_components}...")
        
    #     # Escalar antes de PCA
    #     scaler = StandardScaler()
    #     X_scaled = scaler.fit_transform(X)
        
    #     # Configurar PCA
    #     if n_components is None:
    #         n_components = 0.95  # Valor por defecto
        
    #     pca = PCA(n_components=n_components, random_state=42)
    #     X_pca = pca.fit_transform(X_scaled)
        
    #     # Convertir resultado PCA de vuelta a DataFrame
    #     if isinstance(n_components, float):
    #         n_cols = pca.n_components_
    #     else:
    #         n_cols = n_components
            
    #     X = pd.DataFrame(
    #         X_pca, 
    #         columns=[f'PC{i+1}' for i in range(n_cols)],
    #         index=X.index
    #     )
        
    #     print(f"Varianza explicada total: {sum(pca.explained_variance_ratio_)*100:.2f}%")
        
    #     # Mostrar componentes más importantes
    #     if hasattr(pca, 'explained_variance_ratio_'):
    #         for i, var_exp in enumerate(pca.explained_variance_ratio_[:5]):
    #             print(f"  PC{i+1}: {var_exp*100:.2f}% de varianza")
    
    # Pruebas: Escalado de caracteristicas 
    print("Aplicando RobustScaler...")
    # Guardar los nombres de las columnas antes de la transformación
    original_columns = X.columns
    original_index = X.index

    scaler = RobustScaler()
    X_scaled_array = scaler.fit_transform(X) # Esto devuelve un numpy.ndarray

    # Convertir el numpy.ndarray de vuelta a un DataFrame, manteniendo columnas e índice
    X = pd.DataFrame(X_scaled_array, columns=original_columns, index=original_index)

    # 8. NUEVA: Balanceo de clases mejorado
    print("Aplicando estrategia de balanceo avanzada...")
    
    # Verificar desbalance
    class_counts = y.value_counts().sort_index()
    imbalance_ratio = class_counts.max() / class_counts.min()
    print(f"Ratio de desbalance: {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 1.5:  # Si hay desbalance significativo
        if modo == 'multiclase':
            # Para multiclase, usar ADASYN que es mejor para clases múltiples
            try:
                sampler = ADASYN(random_state=42, n_neighbors=5)
                X_resampled, y_resampled = sampler.fit_resample(X, y)
                
                X = pd.DataFrame(X_resampled, columns=X.columns)
                y = pd.Series(y_resampled)
                
                print("Distribución después de ADASYN:")
                print(y.value_counts().sort_index())
                
            except Exception as e:
                print(f"ADASYN falló: {e}. Usando SMOTE básico.")
                smote = SMOTE(random_state=42, k_neighbors=5)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                X = pd.DataFrame(X_resampled, columns=X.columns)
                y = pd.Series(y_resampled)
        
        elif modo == 'binario':
            # Para binario, usar SMOTETomek (híbrido)
            try:
                sampler = SMOTETomek(random_state=42, smote=SMOTE(k_neighbors=5))
                X_resampled, y_resampled = sampler.fit_resample(X, y)
                
                X = pd.DataFrame(X_resampled, columns=X.columns)
                y = pd.Series(y_resampled)
                
                print("Distribución después de SMOTETomek:")
                print(y.value_counts().sort_index())
                
            except Exception as e:
                print(f"SMOTETomek falló: {e}. Usando SMOTE básico.")
                smote = SMOTE(random_state=42, k_neighbors=5)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                X = pd.DataFrame(X_resampled, columns=X.columns)
                y = pd.Series(y_resampled)
    
    # 9. Validación final
    print(f"\nResumen final:")
    print(f"Características finales: {X.shape[1]}")
    print(f"Muestras finales: {X.shape[0]}")
    print(f"Distribución de clases final:")
    print(y.value_counts().sort_index())
    
    # Asegurar que todas las columnas sean numéricas antes de verificaciones finales
    X = X.select_dtypes(include=[np.number])
    
    # Verificar que no hay valores infinitos o NaN
    if X.isnull().sum().sum() > 0:
        print("Advertencia: Encontrados valores NaN, rellenando con 0")
        X = X.fillna(0)
    
    # Verificar valores infinitos solo en datos numéricos -Probablemente sea mejor eliminar
    try:
        if np.isinf(X.values).sum() > 0:
            print("Advertencia: Encontrados valores infinitos, reemplazando")
            X = X.replace([np.inf, -np.inf], 0)
    except TypeError:
        print("Verificando valores infinitos columna por columna...")
        for col in X.columns:
            if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                if np.isinf(X[col]).sum() > 0:
                    print(f"Reemplazando infinitos en columna: {col}")
                    X[col] = X[col].replace([np.inf, -np.inf], 0)
    
    # Opcional: Guardar datos preparados para análisis
    X.to_csv(f"{TRAIN_TEST_DATA_DIR}/X_{modo}.csv", index=False)
    y.to_csv(f"{TRAIN_TEST_DATA_DIR}/Y_{modo}.csv", index=False)
    
    return X, y
