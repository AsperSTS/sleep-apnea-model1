"""
Módulo para la construcción y entrenamiento de modelos
"""
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from config import *

def construir_modelo(X_train, y_train, X_test, y_test, selected_features=None):
    """
    Construye y entrena el modelo de clasificación para apnea.
    
    Args:
        X_train: DataFrame con características para entrenamiento
        y_train: Series con valores objetivo para entrenamiento
        X_test: DataFrame con características para evaluación
        y_test: Series con valores objetivo para evaluación
        selected_features: Lista de características seleccionadas por preprocessing
    """
    print("Iniciando construcción y entrenamiento de modelos...")
    
    # Si se proporcionan características seleccionadas, usarlas en lugar de las predefinidas
    caracteristicas_numericas = []
    caracteristicas_categoricas = []
    
    if selected_features:
        print(f"Usando características seleccionadas: {selected_features}")
        # Determinar qué características son numéricas y cuáles categóricas basado en el tipo de datos
        for feature in selected_features:
            if feature in X_train.columns:
                if pd.api.types.is_numeric_dtype(X_train[feature]):
                    caracteristicas_numericas.append(feature)
                else:
                    caracteristicas_categoricas.append(feature)
    else:
        # Usar características predefinidas si no se proporcionan selected_features
        caracteristicas_numericas = [col for col in CARACTERISTICAS_NUMERICAS if col in X_train.columns]
        caracteristicas_categoricas = [col for col in CARACTERISTICAS_CATEGORICAS if col in X_train.columns]
    
    print(f"Características numéricas: {caracteristicas_numericas}")
    print(f"Características categóricas: {caracteristicas_categoricas}")
    
    # Definir preprocesadores para características numéricas y categóricas
    transformers = []
    
    if caracteristicas_numericas:
        preprocesador_numerico = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', preprocesador_numerico, caracteristicas_numericas))
    
    if caracteristicas_categoricas:
        preprocesador_categorico = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        transformers.append(('cat', preprocesador_categorico, caracteristicas_categoricas))
    
    # Combinar preprocesadores (o usar 'passthrough' si no hay transformaciones)
    if transformers:
        preprocesador = ColumnTransformer(transformers)
    else:
        preprocesador = 'passthrough'
    
    # Definir modelos candidatos
    modelos = {
        'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE),
        'GradientBoosting': GradientBoostingClassifier(random_state=RANDOM_STATE)
    }
    
    # Parámetros para búsqueda de hiperparámetros
    parametros = {
        'RandomForest': PARAMS_RANDOM_FOREST,
        'GradientBoosting': PARAMS_GRADIENT_BOOSTING
    }
    
    # Diccionario para almacenar resultados
    resultados = {}
    
    # Entrenar y evaluar cada modelo
    for nombre_modelo, modelo in modelos.items():
        print(f"\nEntrenando modelo {nombre_modelo}...")
        # Crear pipeline con preprocesador y modelo
        pipeline = Pipeline([
            ('preprocessor', preprocesador),
            ('classifier', modelo)
        ])
        
        # Búsqueda de hiperparámetros
        grid_search = GridSearchCV(
            pipeline, 
            parametros[nombre_modelo], 
            cv=5, 
            scoring='roc_auc',
            n_jobs=-1
        )
        
        # Entrenar modelo
        grid_search.fit(X_train, y_train)
        
        # Mejor modelo
        mejor_modelo = grid_search.best_estimator_
        
        # Predicciones
        y_pred = mejor_modelo.predict(X_test)
        y_prob = mejor_modelo.predict_proba(X_test)[:, 1]
        
        # Métricas de evaluación
        resultados[nombre_modelo] = {
            'modelo': mejor_modelo,
            'mejores_parametros': grid_search.best_params_,
            'roc_auc': roc_auc_score(y_test, y_prob),
            'reporte_clasificacion': classification_report(y_test, y_pred),
            'matriz_confusion': confusion_matrix(y_test, y_pred)
        }
        
        # Mostrar resultados
        print(f"Resultados para {nombre_modelo}:")
        print(f"ROC AUC: {resultados[nombre_modelo]['roc_auc']:.4f}")
        print("Reporte de clasificación:")
        print(resultados[nombre_modelo]['reporte_clasificacion'])
        
    # Identificar el mejor modelo
    mejor_modelo_nombre = max(resultados, key=lambda x: resultados[x]['roc_auc'])
    print(f"\nEl mejor modelo es: {mejor_modelo_nombre}")
    
    return resultados, mejor_modelo_nombre

# """
# Módulo para la construcción y entrenamiento de modelos
# """
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# from sklearn.impute import SimpleImputer
# from sklearn.model_selection import GridSearchCV
# import pandas as pd
# import numpy as np
# from config import *

# def construir_modelo(X_train, y_train, X_test, y_test):
#     """
#     Construye y entrena el modelo de clasificación para apnea.
#     """
#     print("Iniciando construcción y entrenamiento de modelos...")
    
#     # Definir preprocesadores para características numéricas y categóricas
#     preprocesador_numerico = Pipeline([
#         ('imputer', SimpleImputer(strategy='median')),
#         ('scaler', StandardScaler())
#     ])
    
#     preprocesador_categorico = Pipeline([
#         ('imputer', SimpleImputer(strategy='most_frequent')),
#         ('onehot', OneHotEncoder(handle_unknown='ignore'))
#     ])
    
#     # Combinar preprocesadores
#     preprocesador = ColumnTransformer([
#         ('num', preprocesador_numerico, CARACTERISTICAS_NUMERICAS),
#         ('cat', preprocesador_categorico, CARACTERISTICAS_CATEGORICAS)
#     ])
    
#     # Definir modelos candidatos
#     modelos = {
#         'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE),
#         'GradientBoosting': GradientBoostingClassifier(random_state=RANDOM_STATE)
#     }
    
#     # Parámetros para búsqueda de hiperparámetros
#     parametros = {
#         'RandomForest': PARAMS_RANDOM_FOREST,
#         'GradientBoosting': PARAMS_GRADIENT_BOOSTING
#     }
    
#     # Diccionario para almacenar resultados
#     resultados = {}
    
#     # Entrenar y evaluar cada modelo
#     for nombre_modelo, modelo in modelos.items():
#         print(f"\nEntrenando modelo {nombre_modelo}...")
#         # Crear pipeline con preprocesador y modelo
#         pipeline = Pipeline([
#             ('preprocessor', preprocesador),
#             ('classifier', modelo)
#         ])
        
#         # Búsqueda de hiperparámetros
#         grid_search = GridSearchCV(
#             pipeline, 
#             parametros[nombre_modelo], 
#             cv=5, 
#             scoring='roc_auc',
#             n_jobs=-1
#         )
        
#         # Entrenar modelo
#         grid_search.fit(X_train, y_train)
        
#         # Mejor modelo
#         mejor_modelo = grid_search.best_estimator_
        
#         # Predicciones
#         y_pred = mejor_modelo.predict(X_test)
#         y_prob = mejor_modelo.predict_proba(X_test)[:, 1]
        
#         # Métricas de evaluación
#         resultados[nombre_modelo] = {
#             'modelo': mejor_modelo,
#             'mejores_parametros': grid_search.best_params_,
#             'roc_auc': roc_auc_score(y_test, y_prob),
#             'reporte_clasificacion': classification_report(y_test, y_pred),
#             'matriz_confusion': confusion_matrix(y_test, y_pred)
#         }
        
#         # Mostrar resultados
#         print(f"Resultados para {nombre_modelo}:")
#         print(f"ROC AUC: {resultados[nombre_modelo]['roc_auc']:.4f}")
#         print("Reporte de clasificación:")
#         print(resultados[nombre_modelo]['reporte_clasificacion'])
        
#     # Identificar el mejor modelo
#     mejor_modelo_nombre = max(resultados, key=lambda x: resultados[x]['roc_auc'])
#     print(f"\nEl mejor modelo es: {mejor_modelo_nombre}")
    
#     return resultados, mejor_modelo_nombre