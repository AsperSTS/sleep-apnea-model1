"""
Módulo de análisis exploratorio de datos (EDA) mejorado para dataset SHHS
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy import stats
from config import *

def eda_completo(df):
    """
    Realiza un análisis exploratorio de datos completo para el dataset SHHS.
    """
    print("Iniciando análisis exploratorio de datos completo...")
    
    # Crear directorio para visualizaciones
    import os
    if not os.path.exists('visualizaciones'):
        os.makedirs('visualizaciones')
    
    # 1. Resumen estadístico básico
    print("\n=== Resumen estadístico básico ===")
    resumen = df.describe(include='all').T
    resumen['missing'] = df.isnull().sum()
    resumen['missing_percent'] = (df.isnull().sum() / len(df)) * 100
    print(resumen)
    
    # 2. Análisis de valores faltantes
    print("\n=== Análisis de valores faltantes ===")
    if df.isnull().sum().sum() > 0:  # Si hay valores nulos
        plt.figure(figsize=(12, 8))
        msno.matrix(df)
        plt.title('Patrón de valores faltantes')
        plt.tight_layout()
        plt.savefig('visualizaciones/valores_faltantes_matriz.png')
        
        plt.figure(figsize=(12, 8))
        msno.heatmap(df)
        plt.title('Correlación de valores faltantes')
        plt.tight_layout()
        plt.savefig('visualizaciones/valores_faltantes_correlacion.png')
    else:
        print("No hay valores faltantes para visualizar")
    
    # 3. Distribución de variables clave
    print("\n=== Distribución de variables clave ===")
    variables_numericas = [col for col in df.columns if col.startswith('nsrr_') and 
                         df[col].dtype in ['float64', 'int64'] and 
                         df[col].nunique() > 5]
    
    n_cols = 3
    n_rows = (len(variables_numericas) + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 4 * n_rows))
    for i, var in enumerate(variables_numericas):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(df[var].dropna(), kde=True)
        plt.title(f'Distribución de {var}')
        plt.tight_layout()
    plt.savefig('visualizaciones/distribucion_variables_numericas.png')
    
    # 4. Análisis de outliers
    print("\n=== Análisis de outliers ===")
    plt.figure(figsize=(15, 4 * n_rows))
    for i, var in enumerate(variables_numericas):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.boxplot(y=df[var].dropna())
        plt.title(f'Boxplot de {var}')
        plt.tight_layout()
    plt.savefig('visualizaciones/boxplots_variables_numericas.png')
    
    # Detectar y reportar outliers usando el método IQR
    outliers_summary = {}
    for var in variables_numericas:
        q1 = df[var].quantile(0.25)
        q3 = df[var].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[var] < lower_bound) | (df[var] > upper_bound)][var]
        outliers_summary[var] = {
            'count': len(outliers),
            'percentage': len(outliers) / len(df) * 100,
            'min': outliers.min() if not outliers.empty else None,
            'max': outliers.max() if not outliers.empty else None
        }
    
    outliers_df = pd.DataFrame(outliers_summary).T
    print("Resumen de outliers (método IQR):")
    print(outliers_df)
    
    # 5. Matriz de correlación
    print("\n=== Matriz de correlación ===")
    # Seleccionar solo variables numéricas para la correlación
    df_num = df.select_dtypes(include=['float64', 'int64'])
    
    plt.figure(figsize=(14, 12))
    corr_matrix = df_num.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Matriz de correlación de variables numéricas')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('visualizaciones/matriz_correlacion.png')
    
    # Identificar las correlaciones más fuertes
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.5:  # Umbral de correlación
                corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    corr_pairs_sorted = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)
    print("Correlaciones más fuertes:")
    for var1, var2, corr in corr_pairs_sorted[:10]:  # Top 10
        print(f"{var1} - {var2}: {corr:.4f}")
    
    # 6. Análisis específico de apnea
    print("\n=== Análisis específico de apnea ===")
    
    # Crear variable de severidad de apnea (si no existe)
    if 'apnea_severity' not in df.columns:
        df['apnea_severity'] = pd.cut(
            df['nsrr_ahi_hp3r_aasm15'], 
            bins=[-np.inf, 5, 15, 30, np.inf],  
            labels=['Normal', 'Leve', 'Moderada', 'Severa']
        )
    
    # Distribución de severidad de apnea
    plt.figure(figsize=(10, 6))
    sns.countplot(x='apnea_severity', data=df)
    plt.title('Distribución de severidad de apnea')
    plt.tight_layout()
    plt.savefig('visualizaciones/distribucion_severidad_apnea.png')
    
    # 7. Relación entre variables clínicas y apnea
    print("\n=== Relación entre variables clínicas y apnea ===")
    variables_clinicas = ['nsrr_age', 'nsrr_bmi', 'nsrr_bp_systolic', 'nsrr_bp_diastolic']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, var in enumerate(variables_clinicas):
        sns.boxplot(x='apnea_severity', y=var, data=df, ax=axes[i])
        axes[i].set_title(f'{var} vs Severidad de Apnea')
        
    plt.tight_layout()
    plt.savefig('visualizaciones/variables_clinicas_vs_apnea.png')
    
    # 8. Análisis diferencial entre apnea obstructiva y central
    print("\n=== Análisis diferencial entre apnea obstructiva y central ===")
    
    # Verificar si existen las columnas de clasificación de tipo de apnea
    if 'apnea_central' in df.columns and 'apnea_obstructiva' in df.columns:
        # Características diferenciales
        df_apnea = df[(df['apnea_central'] == 1) | (df['apnea_obstructiva'] == 1)].copy()
        df_apnea['tipo_apnea'] = df_apnea['apnea_central'].map({1: 'Central', 0: 'Obstructiva'})
        
        # Comparar características por tipo de apnea
        variables_comparacion = ['nsrr_age', 'nsrr_bmi', 'nsrr_phrnumar_f1', 
                               'nsrr_ahi_hp3r_aasm15', 'nsrr_ahi_hp4u_aasm15']
        
        fig, axes = plt.subplots(len(variables_comparacion), 1, figsize=(12, 4*len(variables_comparacion)))
        
        for i, var in enumerate(variables_comparacion):
            sns.boxplot(x='tipo_apnea', y=var, data=df_apnea, ax=axes[i])
            axes[i].set_title(f'{var} por tipo de apnea')
            
            # Realizar prueba t para comparar medias
            central = df_apnea[df_apnea['tipo_apnea'] == 'Central'][var].dropna()
            obstructiva = df_apnea[df_apnea['tipo_apnea'] == 'Obstructiva'][var].dropna()
            
            if len(central) > 1 and len(obstructiva) > 1:
                t_stat, p_val = stats.ttest_ind(central, obstructiva, equal_var=False)
                axes[i].annotate(f'p-value: {p_val:.4f}', xy=(0.5, 0.9), xycoords='axes fraction')
        
        plt.tight_layout()
        plt.savefig('visualizaciones/comparacion_tipos_apnea_detallada.png')
        
        # Distribución conjunta de características clave por tipo de apnea
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='nsrr_bmi', y='nsrr_ahi_hp3r_aasm15', 
                      hue='tipo_apnea', data=df_apnea)
        plt.title('BMI vs AHI por tipo de apnea')
        plt.tight_layout()
        plt.savefig('visualizaciones/scatter_bmi_ahi_por_tipo.png')
    
    # 9. Análisis por género
    print("\n=== Análisis por género ===")
    if 'nsrr_sex' in df.columns:
        # Convertir a formato numérico si es necesario
        if df['nsrr_sex'].dtype == 'object':
            gender_map = {'male': 1, 'female': 0}
            df['gender_numeric'] = df['nsrr_sex'].map(gender_map)
        else:
            df['gender_numeric'] = df['nsrr_sex']
            
        gender_labels = {1: 'Male', 0: 'Female'}
        df['gender_label'] = df['gender_numeric'].map(gender_labels)
        
        plt.figure(figsize=(12, 6))
        sns.histplot(x='nsrr_ahi_hp3r_aasm15', hue='gender_label', data=df, kde=True, common_norm=False)
        plt.title('Distribución de AHI por género')
        plt.tight_layout()
        plt.savefig('visualizaciones/ahi_por_genero.png')
        
        plt.figure(figsize=(10, 6))
        crosstab = pd.crosstab(df['gender_label'], df['apnea_severity'])
        crosstab_percent = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
        crosstab_percent.plot(kind='bar', stacked=False)
        plt.title('Distribución porcentual de severidad de apnea por género')
        plt.ylabel('Porcentaje')
        plt.tight_layout()
        plt.savefig('visualizaciones/severidad_apnea_por_genero_percent.png')
    
    # 10. Análisis multivariado de predictores
    print("\n=== Análisis multivariado de predictores ===")
    # PairGrid para visualizar relaciones entre múltiples variables
    predictores_clave = ['nsrr_age', 'nsrr_bmi', 'nsrr_bp_systolic', 'nsrr_ahi_hp3r_aasm15']
    
    if all(pred in df.columns for pred in predictores_clave):
        plt.figure(figsize=(12, 10))
        sns.pairplot(df[predictores_clave + ['apnea_severity']], hue='apnea_severity')
        plt.tight_layout()
        plt.savefig('visualizaciones/pairplot_predictores.png')
    
    return df