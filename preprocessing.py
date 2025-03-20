"""
Módulo mejorado para el preprocesamiento de datos de apnea del sueño
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from config import *

def preprocesar_datos(df, estrategia_imputation='knn'):
    """
    Realiza el preprocesamiento mejorado de los datos.
    
    Args:
        df: DataFrame con los datos crudos
        estrategia_imputation: Estrategia para imputar valores faltantes ('knn', 'median', 'mean')
    
    Returns:
        DataFrame procesado
    """
    print("Iniciando preprocesamiento mejorado de datos...")
    # Hacer una copia para no modificar el original
    df_procesado = df.copy()
    
    # 1. Limpieza básica de datos
    print("Realizando limpieza básica de datos...")
    
    # Convertir variables categóricas a formato adecuado
    variables_categoricas = ['nsrr_sex', 'nsrr_race', 'nsrr_ethnicity', 
                            'nsrr_current_smoker', 'nsrr_ever_smoker']
    
    for var in variables_categoricas:
        if var in df_procesado.columns:
            # Si la variable ya es numérica, saltamos
            if pd.api.types.is_numeric_dtype(df_procesado[var]):
                continue
                
            # Convertir a minúsculas y eliminar espacios
            if df_procesado[var].dtype == 'object':
                df_procesado[var] = df_procesado[var].str.lower() if hasattr(df_procesado[var], 'str') else df_procesado[var]
            
            # Mapeo específico para cada variable
            if var == 'nsrr_sex':
                df_procesado[var] = df_procesado[var].map({'male': 1, 'female': 0})
            elif var in ['nsrr_current_smoker', 'nsrr_ever_smoker']:
                df_procesado[var] = df_procesado[var].map({'yes': 1, 'no': 0, 'not reported': np.nan})
    
    # 2. Manejo avanzado de valores faltantes
    print(f"Aplicando estrategia de imputación: {estrategia_imputation}...")
    
    # Separar variables numéricas y categóricas para imputación
    variables_numericas = df_procesado.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Imputación para variables numéricas
    if estrategia_imputation == 'knn':
        # KNN Imputer para variables numéricas
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        df_procesado[variables_numericas] = imputer.fit_transform(df_procesado[variables_numericas])
    else:
        # Imputación simple para variables numéricas
        imputer = SimpleImputer(strategy=estrategia_imputation)
        df_procesado[variables_numericas] = imputer.fit_transform(df_procesado[variables_numericas])
    
    # Imputación para variables categóricas (moda)
    for var in variables_categoricas:
        if var in df_procesado.columns and df_procesado[var].isnull().sum() > 0:
            moda = df_procesado[var].mode()[0]
            df_procesado.loc[:, var] = df_procesado[var].fillna(moda)
    
    # 3. Detección y manejo de outliers
    print("Detectando y manejando outliers...")
    for var in variables_numericas:
        # Calcular límites para outliers usando el método IQR
        q1 = df_procesado[var].quantile(0.25)
        q3 = df_procesado[var].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr  # Más permisivo que el estándar 1.5*IQR
        upper_bound = q3 + 3 * iqr
        
        # Identificar outliers extremos
        extremos = df_procesado[(df_procesado[var] < lower_bound) | (df_procesado[var] > upper_bound)][var]
        
        if len(extremos) > 0:
            # Aplicar técnica de capping para limitar valores extremos
            df_procesado[var] = df_procesado[var].clip(lower=lower_bound, upper=upper_bound)
            print(f"  {var}: {len(extremos)} outliers limitados")
    
    # 4. Filtrar por criterios médicos
    print("Aplicando filtros basados en criterios médicos...")
    # Filtro de edad (ajustable según población objetivo)
    if 'nsrr_age' in df_procesado.columns:
        df_procesado = df_procesado[(df_procesado['nsrr_age'] >= EDAD_MIN) & 
                                  (df_procesado['nsrr_age'] <= EDAD_MAX)]
    
    # 5. Creación avanzada de características
    print("Creando características adicionales...")
    
    # Índice de masa corporal categorizado según OMS
    if 'nsrr_bmi' in df_procesado.columns:
        df_procesado['bmi_categoria'] = pd.cut(
            df_procesado['nsrr_bmi'], 
            bins=[0, 18.5, 25, 30, 35, 40, float('inf')],
            labels=['Bajo peso', 'Normal', 'Sobrepeso', 'Obesidad I', 'Obesidad II', 'Obesidad III']
        )
    
    # Categoría de presión arterial (basado en guías JNC 7)
    if 'nsrr_bp_systolic' in df_procesado.columns and 'nsrr_bp_diastolic' in df_procesado.columns:
        # Función para categorizar PA
        def categorizar_pa(row):
            sys = row['nsrr_bp_systolic']
            dia = row['nsrr_bp_diastolic']
            
            if sys < 120 and dia < 80:
                return 'Normal'
            elif (sys >= 120 and sys < 140) or (dia >= 80 and dia < 90):
                return 'Prehipertensión'
            elif (sys >= 140 and sys < 160) or (dia >= 90 and dia < 100):
                return 'HTA Estadio 1'
            elif sys >= 160 or dia >= 100:
                return 'HTA Estadio 2'
            else:
                return np.nan
        
        df_procesado['categoria_pa'] = df_procesado.apply(categorizar_pa, axis=1)
    
    # 6. Variables objetivo mejoradas
    print("Creando variables objetivo mejoradas...")
    
    # Clasificación de severidad usando criterios AASM
    if 'nsrr_ahi_hp3r_aasm15' in df_procesado.columns:
        df_procesado['apnea_severity'] = pd.cut(
            df_procesado['nsrr_ahi_hp3r_aasm15'], 
            bins=[-np.inf, 5, 15, 30, np.inf],  
            labels=['Normal', 'Leve', 'Moderada', 'Severa']
        )
        
        # Variable binaria de apnea (AHI ≥ 5)
        df_procesado['apnea'] = (df_procesado['nsrr_ahi_hp3r_aasm15'] >= 5).astype(int)
    
    # 7. Clasificación mejorada de tipo de apnea
    print("Mejorando clasificación de tipos de apnea...")
    
    """
    Criterios mejorados para distinción entre apnea central y obstructiva:
    
    1. AHI: Índice base de apnea-hipopnea
    2. CAI: Índice de apneas centrales (estimado a partir de relaciones entre índices)
    3. OAI: Índice de apneas obstructivas (estimado)
    4. Desaturación: Patrón de desaturación de oxígeno
    5. Arousal: Índice de microdespertares
    
    La literatura médica sugiere que la apnea central tiende a tener:
    - Mayor índice de desaturaciones profundas (≥4%)
    - Menor índice de microdespertares
    - Patrón respiratorio más regular durante eventos
    """
    
    # Mejorar estimación del componente central vs obstructivo
    if all(col in df_procesado.columns for col in ['nsrr_ahi_hp3r_aasm15', 'nsrr_ahi_hp4u_aasm15', 'nsrr_phrnumar_f1']):
        # Índice estimado de apneas/hipopneas con desaturación significativa (≥4%)
        df_procesado['indice_desaturacion'] = df_procesado['nsrr_ahi_hp4u_aasm15'] / (df_procesado['nsrr_ahi_hp3r_aasm15'] + 0.1)
        
        # Índice de arousal normalizado
        max_arousal = df_procesado['nsrr_phrnumar_f1'].max()
        df_procesado['arousal_norm'] = df_procesado['nsrr_phrnumar_f1'] / max_arousal if max_arousal > 0 else 0
        
        # Algoritmo de clasificación más sofisticado basado en la literatura
        # Definir umbrales basados en distribuciones
        umbral_desaturacion = df_procesado['indice_desaturacion'].quantile(0.7)  # Percentil 70
        umbral_arousal = df_procesado['arousal_norm'].quantile(0.3)  # Percentil 30
        
        # Clasificación
        condiciones = [
            # No apnea
            (df_procesado['apnea'] == 0),
            # Apnea predominantemente central
            ((df_procesado['apnea'] == 1) & 
             (df_procesado['indice_desaturacion'] > umbral_desaturacion) & 
             (df_procesado['arousal_norm'] < umbral_arousal)),
            # Apnea predominantemente obstructiva (por defecto)
            (df_procesado['apnea'] == 1)
        ]
        
        opciones = [0, 1, 2]  # 0=No apnea, 1=Central, 2=Obstructiva
        df_procesado['tipo_apnea'] = np.select(condiciones, opciones, default=np.nan)
        
        # Variables binarias
        df_procesado['apnea_central'] = (df_procesado['tipo_apnea'] == 1).astype(int)
        df_procesado['apnea_obstructiva'] = (df_procesado['tipo_apnea'] == 2).astype(int)
        
        # Verificar distribución resultante
        print("Distribución de tipos de apnea:")
        print(df_procesado['tipo_apnea'].map({0: 'No apnea', 1: 'Central', 2: 'Obstructiva'}).value_counts())
    
    print(f"Preprocesamiento completado. Registros resultantes: {len(df_procesado)}")
    return df_procesado

def seleccionar_caracteristicas(X, y, metodo='univariado', k=10):
    """
    Selecciona las características más relevantes para el modelo.
    
    Args:
        X: DataFrame con variables predictoras
        y: Series con variable objetivo
        metodo: Método de selección ('univariado', 'rfe', 'importancia')
        k: Número de características a seleccionar
    
    Returns:
        X_selected: DataFrame con características seleccionadas
        selected_features: Lista de nombres de características seleccionadas
    """
    print(f"Seleccionando características con método {metodo}...")
    
    if metodo == 'univariado':
        # Selección univariada basada en pruebas F
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Obtener nombres de características seleccionadas
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
    elif metodo == 'rfe':
        # Recursive Feature Elimination con Random Forest
        estimator = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        selector = RFE(estimator, n_features_to_select=k, step=1)
        X_selected = selector.fit_transform(X, y)
        
        # Obtener nombres de características seleccionadas
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
    elif metodo == 'importancia':
        # Importancia de características con Random Forest
        estimator = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        estimator.fit(X, y)
        
        # Obtener y ordenar importancias
        importances = pd.Series(estimator.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False)
        
        # Seleccionar top k características
        selected_features = importances.index[:k].tolist()
        X_selected = X[selected_features]
    
    else:
        raise ValueError(f"Método de selección no reconocido: {metodo}")
    
    print(f"Características seleccionadas ({len(selected_features)}):")
    for feat in selected_features:
        print(f"  - {feat}")
    
    return X_selected, selected_features

def preparar_datos_modelo(df, solo_obstructiva=True, balanceo_clases='smote', 
                         metodo_seleccion='importancia', num_caracteristicas=10):
    """
    Prepara los datos para el entrenamiento del modelo con opciones avanzadas.
    """
    print(f"Preparando datos para modelado de apnea...")
    
    # 1. Filtrar según el tipo de apnea
    if solo_obstructiva:
        # Apnea obstructiva
        y = df['apnea_obstructiva']
    else:  # 'cualquiera'
        # Apnea de cualquier tipo
        y = df['apnea']
    
    # 2. Preparar variables predictoras
    # Eliminar variables objetivo y otras que no deben incluirse
    variables_a_excluir = ['apnea', 'apnea_obstructiva', 'apnea_central', 
                          'tipo_apnea', 'apnea_severity']
    
    # Añadir columnas categóricas que necesitan convertirse a dummies
    variables_categoricas = ['bmi_categoria', 'categoria_pa']
    
    # También incluir otras variables categóricas que puedan quedar
    for col in df.columns:
        if df[col].dtype == 'object' and col not in variables_categoricas:
            variables_categoricas.append(col)
    
    # Crear copias para evitar modificar el dataframe original
    X = df.copy()
    
    # Eliminar variables que no deben incluirse
    for var in variables_a_excluir:
        if var in X.columns:
            X = X.drop(var, axis=1)
    
    # Convertir TODAS las variables categóricas a dummies
    for var in variables_categoricas:
        if var in X.columns:
            dummies = pd.get_dummies(X[var], prefix=var, drop_first=True)
            X = pd.concat([X, dummies], axis=1)
            X = X.drop(var, axis=1)
    
    # Verificar que no queden columnas no numéricas
    non_numeric_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
    if non_numeric_cols:
        print(f"Advertencia: Eliminando columnas no numéricas restantes: {non_numeric_cols}")
        X = X.drop(columns=non_numeric_cols)
    
    # 3. Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # 4. Aplicar escalado a características numéricas
    scaler = RobustScaler()  # Más robusto a outliers que StandardScaler
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # 5. Seleccionar características
    X_train_selected, selected_features = seleccionar_caracteristicas(
        X_train, y_train, metodo=metodo_seleccion, k=num_caracteristicas
    )
    
    # Aplicar selección de características al conjunto de prueba
    X_test_selected = X_test[selected_features]
    
    # 6. Balancear clases si es necesario
    if balanceo_clases != 'none':
        print(f"Aplicando balanceo de clases con método: {balanceo_clases}")
        
        if balanceo_clases == 'smote':
            # Sobremuestreo con SMOTE
            smote = SMOTE(random_state=RANDOM_STATE)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)
            
        elif balanceo_clases == 'undersample':
            # Submuestreo aleatorio
            rus = RandomUnderSampler(random_state=RANDOM_STATE)
            X_train_resampled, y_train_resampled = rus.fit_resample(X_train_selected, y_train)
            
        elif balanceo_clases == 'combined':
            # Combinación de sobremuestreo y submuestreo
            ratio_under = 0.5  # Reducir clase mayoritaria al 50%
            rus = RandomUnderSampler(sampling_strategy=ratio_under, random_state=RANDOM_STATE)
            X_temp, y_temp = rus.fit_resample(X_train_selected, y_train)
            
            # Luego aplicar SMOTE
            smote = SMOTE(random_state=RANDOM_STATE)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_temp, y_temp)
            
        else:
            raise ValueError(f"Método de balanceo no reconocido: {balanceo_clases}")
            
        # Actualizar los conjuntos de entrenamiento balanceados
        X_train_final = X_train_resampled
        y_train_final = y_train_resampled
        
        # Imprimir distribución de clases después del balanceo
        print("Distribución de clases después del balanceo:")
        print(pd.Series(y_train_final).value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))
    else:
        # Sin balanceo de clases
        X_train_final = X_train_selected
        y_train_final = y_train
    
    # 7. Imprimir información sobre los conjuntos de datos
    print(f"Dimensiones del conjunto de entrenamiento: {X_train_final.shape}")
    print(f"Dimensiones del conjunto de prueba: {X_test_selected.shape}")
    print(f"Distribución de clases en entrenamiento: {pd.Series(y_train_final).value_counts()}")
    print(f"Distribución de clases en prueba: {pd.Series(y_test).value_counts()}")
    
    return X_train_final, X_test_selected, y_train_final, y_test, selected_features


# """
# Módulo para el preprocesamiento de datos
# """
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from config import *

# def preprocesar_datos(df):
#     """
#     Realiza el preprocesamiento inicial de los datos.
#     """
#     print("Iniciando preprocesamiento de datos...")
#     # Hacer una copia para no modificar el original
#     df_procesado = df.copy()
    
#     # Filtrar por rango de edad (18-60 años)
#     print(f"Filtrando pacientes con edades entre {EDAD_MIN} y {EDAD_MAX} años...")
#     df_procesado = df_procesado[(df_procesado['nsrr_age'] >= EDAD_MIN) & (df_procesado['nsrr_age'] <= EDAD_MAX)]
    
#     # Convertir variables categóricas
#     print("Convirtiendo variables categóricas...")
#     df_procesado['nsrr_sex'] = df_procesado['nsrr_sex'].map({'male': 1, 'female': 0})
#     df_procesado['nsrr_current_smoker'] = df_procesado['nsrr_current_smoker'].map({'yes': 1, 'no': 0})
#     df_procesado['nsrr_ever_smoker'] = df_procesado['nsrr_ever_smoker'].map({'yes': 1, 'no': 0})
    
#     # Eliminar filas con valores nulos en características clave
#     print("Manejando valores nulos...")
#     columnas_clave = ['nsrr_age', 'nsrr_bmi', 'nsrr_ahi_hp3r_aasm15']
#     df_procesado = df_procesado.dropna(subset=columnas_clave)
    
#     # Crear variable objetivo: definir apnea basado en AHI
#     print("Creando variables objetivo...")
#     # Usando nsrr_ahi_hp3r_aasm15 como indicador principal (criterio AASM 2015)
#     df_procesado['apnea_severity'] = pd.cut(
#         df_procesado['nsrr_ahi_hp3r_aasm15'], 
#         bins=[-np.inf, 5, 15, 30, np.inf],  
#         labels=['Normal', 'Leve', 'Moderada', 'Severa'],
#         right=False  # Cerrado a la izquierda, abierto a la derecha
#     )

#     """
#         Normal	AHI < 5
#         Leve	5 ≤ AHI < 15
#         Moderada	15 ≤ AHI < 30
#         Severa	AHI ≥ 30
#     """
#     # Crear una variable binaria de apnea (AHI ≥ 5)
#     df_procesado['apnea'] = (df_procesado['nsrr_ahi_hp3r_aasm15'] >= 5).astype(int)
    
#     # Identificación de apnea central vs obstructiva
#     # Clasificación basada en criterios de expertos:
#     # - Mayor proporción de apneas con desaturación significativa (≥4%) 
#     #   respecto a las apneas totales puede indicar componente central
#     # - Mayor índice de arousal puede indicar componente obstructivo
#     print("Clasificando tipos de apnea...")
    
#     # Calcular proporción entre diferentes índices AHI para estimar componente central
#     # (Esta es una aproximación y debe ser validada clínicamente)
#     df_procesado['proporcion_desaturacion'] = (df_procesado['nsrr_ahi_hp4u_aasm15'] / 
#                                               (df_procesado['nsrr_ahi_hp3r_aasm15'] + 0.1))
    
#     # Normalizar índice de arousal para comparación
#     df_procesado['arousal_normalizado'] = df_procesado['nsrr_phrnumar_f1'] / df_procesado['nsrr_phrnumar_f1'].max()
    
#     # Algoritmo simple para clasificar tipo de apnea
#     # Valores altos de proporción y bajos de arousal sugieren componente central
#     # Este umbral es arbitrario y debe ajustarse con validación clínica
#     df_procesado['apnea_central'] = ((df_procesado['proporcion_desaturacion'] > 0.75) & 
#                                     (df_procesado['arousal_normalizado'] < 0.5) & 
#                                     (df_procesado['apnea'] == 1)).astype(int)
    
#     df_procesado['apnea_obstructiva'] = ((df_procesado['apnea'] == 1) & 
#                                         (df_procesado['apnea_central'] == 0)).astype(int)
    
    
#     print(f"Preprocesamiento completado. Registros resultantes: {len(df_procesado)}")
#     return df_procesado

# def analisis_exploratorio(df):
#     """
#     Realiza un análisis exploratorio básico de los datos.
#     """
#     print("Realizando análisis exploratorio...")
    
#     # Crear directorio para visualizaciones
#     import os
#     if not os.path.exists('visualizaciones'):
#         os.makedirs('visualizaciones')
    
#     # Resumen estadístico
#     print("Resumen estadístico:")
#     print(df.describe())
    
#     # Contar valores nulos
#     print("\nValores nulos por columna:")
#     print(df.isnull().sum())
    
#     # Distribución de la variable objetivo
#     print("\nDistribución de severidad de apnea:")
#     print(df['apnea_severity'].value_counts())
    
#     # Distribución de tipo de apnea
#     print("\nDistribución de tipo de apnea:")
#     tipo_apnea = pd.DataFrame({
#         'Normal': (df['apnea'] == 0).sum(),
#         'Apnea Obstructiva': df['apnea_obstructiva'].sum(),
#         'Apnea Central': df['apnea_central'].sum()
#     }, index=['Conteo'])
#     print(tipo_apnea)
    
#     # Visualizaciones
#     plt.figure(figsize=(12, 10))
    
#     # Distribución de AHI
#     plt.subplot(2, 2, 1)
#     sns.histplot(df['nsrr_ahi_hp3r_aasm15'], kde=True)
#     plt.title('Distribución de AHI')
    
#     # BMI vs AHI
#     plt.subplot(2, 2, 2)
#     sns.scatterplot(x='nsrr_bmi', y='nsrr_ahi_hp3r_aasm15', hue='apnea_severity', data=df)
#     plt.title('BMI vs AHI')
    
#     # Edad vs AHI
#     plt.subplot(2, 2, 3)
#     sns.boxplot(x='apnea_severity', y='nsrr_age', data=df)
#     plt.title('Edad vs Severidad de Apnea')
    
#     # Distribución por género
#     plt.subplot(2, 2, 4)
#     gender_apnea = pd.crosstab(df['nsrr_sex'], df['apnea_severity'])
#     gender_apnea.plot(kind='bar', stacked=True)
#     plt.title('Distribución de Apnea por Género')
#     plt.tight_layout()
#     plt.savefig('visualizaciones/analisis_exploratorio.png')
    
#     # Visualización específica para apnea obstructiva vs central
#     plt.figure(figsize=(10, 8))
    
#     # BMI por tipo de apnea
#     plt.subplot(2, 1, 1)
#     sns.boxplot(x=df['apnea_severity'], 
#                 y=df['nsrr_bmi'], 
#                 hue=df['apnea_central'].map({0: 'Obstructiva/Normal', 1: 'Central'}))
#     plt.title('BMI por Tipo de Apnea')
    
#     # Índice de arousal por tipo de apnea
#     plt.subplot(2, 1, 2)
#     sns.boxplot(x=df['apnea_severity'], 
#                 y=df['nsrr_phrnumar_f1'], 
#                 hue=df['apnea_central'].map({0: 'Obstructiva/Normal', 1: 'Central'}))
#     plt.title('Índice de Arousal por Tipo de Apnea')
    
#     plt.tight_layout()
#     plt.savefig('visualizaciones/comparacion_tipos_apnea.png')
    
#     return df

# def preparar_datos_modelo(df, solo_obstructiva=True):
#     """
#     Prepara los datos para el entrenamiento del modelo.
#     """
#     print("Preparando datos para el modelado...")
    
#     # Si solo queremos enfocarnos en apnea obstructiva
#     if solo_obstructiva:
#         # Excluir pacientes con apnea central
#         df_filtrado = df[df['apnea_central'] == 0]
#         y = df_filtrado['apnea_obstructiva']
#         print(f"Filtrando solo para apnea obstructiva. Registros: {len(df_filtrado)}")
#     else:
#         # Usar todos los pacientes
#         df_filtrado = df
#         y = df_filtrado['apnea']
#         print(f"Usando todos los registros para clasificación general de apnea. Registros: {len(df_filtrado)}")
    
#     # Definir X según las características configuradas
#     X = df_filtrado[CARACTERISTICAS_NUMERICAS + CARACTERISTICAS_CATEGORICAS]
    
#     # Dividir en conjuntos de entrenamiento y prueba
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
#     )
    
#     print(f"Datos divididos: {len(X_train)} para entrenamiento, {len(X_test)} para prueba")
#     return X_train, X_test, y_train, y_test