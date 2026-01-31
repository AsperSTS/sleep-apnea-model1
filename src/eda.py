"""
Exploratory Data Analysis (EDA) module for the SHHS dataset.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy import stats
import os
import logging
from datetime import datetime
from stadistic_analysis import *

def EDA(df, eda_dir: str, log_dir: str = None):
    """
    Executes the full Exploratory Data Analysis pipeline.
    
    Generates statistical summaries, missing value analysis, distribution plots, 
    and specific apnea-related visualizations. Artifacts are saved to `eda_dir`.
    """
    
    # --- Comparison Table Configuration ---

    features_for_table = [
        'nsrr_age', 'nsrr_sex', 'nsrr_bmi', 'nsrr_bp_systolic', 
        'nsrr_bp_diastolic', 'nsrr_current_smoker', 'nsrr_ever_smoker', 'nsrr_race', 'nsrr_ethnicity'
    ]
    
    # Define Apnea-Hypopnea Index (AHI) thresholds
    ahi_cutoffs = {
        'AHI < 5': (5, 'lt', 'AHI < 5'),
        'AHI ≥ 5': (5, 'ge', 'AHI ≥ 5'),
    }

    # Verify features exist in dataframe before processing
    existing_features = [f for f in features_for_table if f in df.columns]
    
    if 'nsrr_ahi_hp3r_aasm15' in df.columns:
        generate_comparison_table(
            df=df,
            feature_columns=existing_features,
            category_column='nsrr_ahi_hp3r_aasm15',
            category_cutoffs=ahi_cutoffs,
            eda_dir=eda_dir
        )
    
    # Ensure output directory exists
    if not os.path.exists(f'{eda_dir}'):
        os.makedirs(f'{eda_dir}')
    
    # --- Logging Setup ---

    if log_dir is None:
        log_dir = f"eda_analysis.log"
    
    log_path = os.path.join(eda_dir, log_dir)
    
    logger = logging.getLogger('EDA_Analysis')
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate log entries by clearing existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info("Starting EDA pipeline...")
    
    # 1. Statistical Summary & Nulls
    logger.info("\n=== Statistical Summary ===")
    resumen = df.describe(include='all').T
    resumen['missing'] = df.isnull().sum()
    resumen['missing_percent'] = (df.isnull().sum() / len(df)) * 100
    logger.info(f"Stats:\n{resumen}")
    
    # 2. Missing Value Analysis
    logger.info("\n=== Missing Values Analysis ===")
    
    plt.figure(figsize=(12, 8))
    msno.matrix(df)
    plt.title('Missing Value Patterns')
    plt.savefig(f'{eda_dir}/valores_faltantes_matriz.png')
    plt.close()
    
    # Generate correlation heatmap only if multiple columns have missing data
    cols_with_missing = df.columns[df.isnull().any()].tolist()
    if len(cols_with_missing) > 1:
        plt.figure(figsize=(12, 8))
        msno.heatmap(df)
        plt.title('Missing Value Correlation')
        plt.savefig(f'{eda_dir}/valores_faltantes_correlacion.png')
        plt.close()
    
    # 3. Numeric Distributions (nsrr_ prefix)
    logger.info("\n=== Variable Distributions ===")
    variables_numericas = [col for col in df.columns if col.startswith('nsrr_') and 
                           df[col].dtype in ['float64', 'int64'] and 
                           df[col].nunique() > 5]
    
    n_cols = 3
    n_rows = (len(variables_numericas) + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 4 * n_rows))
    for i, var in enumerate(variables_numericas):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(df[var].dropna(), kde=True)
        plt.title(f'Distribution: {var}')
        plt.tight_layout()
    plt.savefig(f'{eda_dir}/distribucion_variables_numericas.png')
    plt.close()
    
    # 4. Outlier Analysis (IQR Method)
    logger.info("\n=== Outlier Analysis ===")
    plt.figure(figsize=(15, 4 * n_rows))
    for i, var in enumerate(variables_numericas):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.boxplot(y=df[var].dropna())
        plt.title(f'Boxplot: {var}')
        plt.tight_layout()
    plt.savefig(f'{eda_dir}/boxplots_variables_numericas.png')
    plt.close()
    
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
    
    logger.info(f"Outlier Summary:\n{pd.DataFrame(outliers_summary).T}")
    
    # 5. Correlation Matrix
    logger.info("\n=== Correlation Matrix ===")
    df_num = df.select_dtypes(include=['float64', 'int64'])
    
    plt.figure(figsize=(14, 12))
    corr_matrix = df_num.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{eda_dir}/matriz_correlacion.png')
    plt.close()
    
    # Log top 10 strong correlations (> 0.5)
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.5:
                corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    corr_pairs_sorted = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)
    for var1, var2, corr in corr_pairs_sorted[:10]:
        logger.info(f"{var1} - {var2}: {corr:.4f}")
    
    # 6. Apnea Severity Analysis
    logger.info("\n=== Apnea Severity Analysis ===")
    if 'apnea_severity' not in df.columns:
        df['apnea_severity'] = pd.cut(
            df['nsrr_ahi_hp3r_aasm15'], 
            bins=[-np.inf, 5, 15, 30, np.inf],  
            labels=['Normal', 'Leve', 'Moderada', 'Severa']
        )
    
    plt.figure(figsize=(10, 6))
    sns.countplot(x='apnea_severity', data=df)
    plt.tight_layout()
    plt.savefig(f'{eda_dir}/distribucion_severidad_apnea.png')
    plt.close()
    
    # 7. Clinical Variables vs Apnea Severity
    variables_clinicas = ['nsrr_age', 'nsrr_bmi', 'nsrr_bp_systolic', 'nsrr_bp_diastolic']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, var in enumerate(variables_clinicas):
        sns.boxplot(x='apnea_severity', y=var, data=df, ax=axes[i])
        axes[i].set_title(f'{var} vs Severity')
        
    plt.tight_layout()
    plt.savefig(f'{eda_dir}/variables_clinicas_vs_apnea.png')
    plt.close()
    
    # 8. Comparison: Obstructive vs Central Apnea
    if 'apnea_central' in df.columns and 'apnea_obstructiva' in df.columns:
        logger.info("\n=== Differential: Central vs Obstructive ===")
        df_apnea = df[(df['apnea_central'] == 1) | (df['apnea_obstructiva'] == 1)].copy()
        df_apnea['tipo_apnea'] = df_apnea['apnea_central'].map({1: 'Central', 0: 'Obstructiva'})
        
        variables_comparacion = ['nsrr_age', 'nsrr_bmi', 'nsrr_ahi_hp3r_aasm15']
        fig, axes = plt.subplots(len(variables_comparacion), 1, figsize=(12, 4*len(variables_comparacion)))
        
        for i, var in enumerate(variables_comparacion):
            sns.boxplot(x='tipo_apnea', y=var, data=df_apnea, ax=axes[i])
            
            # T-test to check for statistical significance
            central = df_apnea[df_apnea['tipo_apnea'] == 'Central'][var].dropna()
            obstructiva = df_apnea[df_apnea['tipo_apnea'] == 'Obstructiva'][var].dropna()
            
            if len(central) > 1 and len(obstructiva) > 1:
                t_stat, p_val = stats.ttest_ind(central, obstructiva, equal_var=False)
                axes[i].annotate(f'p-value: {p_val:.4f}', xy=(0.5, 0.9), xycoords='axes fraction')
        
        plt.tight_layout()
        plt.savefig(f'{eda_dir}/comparacion_tipos_apnea_detallada.png')
        plt.close()
        
    # 9. Gender Analysis
    if 'nsrr_sex' in df.columns:
        logger.info("\n=== Gender Analysis ===")
        # Normalize gender labels
        if df['nsrr_sex'].dtype == 'object':
            gender_map = {'male': 1, 'female': 0}
            df['gender_numeric'] = df['nsrr_sex'].map(gender_map)
        else:
            df['gender_numeric'] = df['nsrr_sex']
            
        df['gender_label'] = df['gender_numeric'].map({1: 'Male', 0: 'Female'})
        
        # AHI distribution by gender
        plt.figure(figsize=(12, 6))
        sns.histplot(x='nsrr_ahi_hp3r_aasm15', hue='gender_label', data=df, kde=True, common_norm=False)
        plt.savefig(f'{eda_dir}/ahi_por_genero.png')
        plt.close()
        
    # 10. Key Predictor Pairplot
    predictores_clave = ['nsrr_age', 'nsrr_bmi', 'nsrr_bp_systolic', 'nsrr_ahi_hp3r_aasm15']
    if all(pred in df.columns for pred in predictores_clave):
        sns.pairplot(df[predictores_clave + ['apnea_severity']], hue='apnea_severity')
        plt.savefig(f'{eda_dir}/pairplot_predictores.png')
        plt.close()
    
    logger.info("EDA completed. Artifacts saved.")
    file_handler.close()
    logger.removeHandler(file_handler)