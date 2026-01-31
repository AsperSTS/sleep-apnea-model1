# Sleep Apnea Diagnosis

Machine learning models for diagnosing obstructive and central sleep apnea using SHHS and HCHS datasets.

## English Version

### Overview

This project builds ML models to diagnose sleep apnea using clinical data from the Sleep Heart Health Study (SHHS) and Hispanic Community Health Study (HCHS), both available at SleepData.org. The models predict binary classification (apnea vs. normal) and multi-class severity levels.

### Goals

- Build accurate ML models for sleep apnea diagnosis using non-invasive clinical data
- Compare SVM, Random Forest, and Gradient Boosting performance
- Handle class imbalance in medical datasets
- Generate interpretable results for clinical use

### Methods

**Data Processing:**
- KNN imputation for missing values
- 3-IQR outlier detection (conservative approach for medical data)
- SMOTE/ADASYN for class balancing
- Feature selection using mutual information and F-scores

**Feature Engineering:**
- BMI categories (WHO standards)
- Blood pressure levels (AHA guidelines)
- Obesity indicators (BMI ≥30, ≥35)
- Hypertension severity
- Combined risk scores

### Project Structure

```
sleep-apnea-diagnosis/
├── test/
├── src/
│   ├── train.py             # Training script
│   ├── preprocessing.py     # Data preprocessing
│   ├── eda.py               # Exploratory analysis
│   ├── prepare_data.py      # Data preparation
│   ├── predict.py           # Prediction script
│   ├── config.py            # Configuration
│   ├── utils.py             # Utilities
│   ├── svm.py               # SVM model
│   ├── random_forest.py     # Random Forest model
│   ├── gradient_boost.py    # Gradient Boosting model
│   ├── models/              # Saved models
│   ├── reports/             # Reports and plots
│   ├── train_test/          # Train/test splits
│   ├── visual_eda/          # EDA plots
│   ├── visual_pre/          # Preprocessing plots
│   └── visual_model/        # Model visualizations
```

### Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn missingno
```

### Usage

**Run EDA only (no training):**
```bash
python train.py --eda
```

**Train binary classifier:**
```bash
python train.py --train --modo binario
```

**Train multi-class classifier:**
```bash
python train.py --train --modo multiclase
```

### Features

**Clinical inputs:**
- Age, BMI, blood pressure (systolic/diastolic)
- Smoking status (current/former)
- Gender, race/ethnicity
- Arousal index and polysomnographic measures

**Engineered features:**
- BMI categories
- Blood pressure classification
- Obesity flags
- Hypertension levels
- Clinical risk combinations

### Models

Three algorithms available:
- **SVM**: Good accuracy with tuned hyperparameters
- **Random Forest**: Best for feature importance analysis
- **Gradient Boosting**: Handles imbalanced data well

Classification modes:
- **Binary**: Normal vs. Sleep Apnea (AHI ≥ 5)
- **Multi-class**: Normal, Mild, Moderate, Severe (AHI thresholds: 5, 15, 30)

### Output

The pipeline generates:
- Confusion matrices
- Classification reports
- Feature importance plots
- ROC and precision-recall curves
- Distribution plots
- Correlation heatmaps

### Data Sources

- **SHHS**: Multi-site cohort study on sleep disorders
- **HCHS**: Hispanic/Latino population focus
- **Source**: SleepData.org (National Sleep Research Resource)

### Disclaimer

This is a research tool. Do not use for medical diagnosis without clinical validation. Models trained on specific populations may not generalize to all demographics.

---

## Versión en Español

### Descripción

Este proyecto construye modelos de ML para diagnosticar apnea del sueño usando datos clínicos de SHHS y HCHS (disponibles en SleepData.org). Los modelos predicen clasificación binaria (apnea vs. normal) y niveles de severidad multi-clase.

### Objetivos

- Construir modelos ML precisos para diagnóstico de apnea del sueño con datos clínicos no invasivos
- Comparar rendimiento de SVM, Random Forest y Gradient Boosting
- Manejar desbalance de clases en datasets médicos
- Generar resultados interpretables para uso clínico

### Métodos

**Procesamiento de datos:**
- Imputación KNN para valores faltantes
- Detección de outliers con 3-IQR (conservador para datos médicos)
- SMOTE/ADASYN para balanceo de clases
- Selección de features con información mutua y F-scores

**Ingeniería de features:**
- Categorías de IMC (estándares OMS)
- Niveles de presión arterial (guías AHA)
- Indicadores de obesidad (IMC ≥30, ≥35)
- Severidad de hipertensión
- Puntajes de riesgo combinados

### Instalación

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn missingno
```

### Uso

**Solo análisis exploratorio (sin entrenamiento):**
```bash
python train.py --eda
```

**Entrenar clasificador binario:**
```bash
python train.py --train --modo binario
```

**Entrenar clasificador multi-clase:**
```bash
python train.py --train --modo multiclase
```

### Features

**Inputs clínicos:**
- Edad, IMC, presión arterial (sistólica/diastólica)
- Estado de fumador (actual/anterior)
- Género, raza/etnia
- Índice de arousal y medidas polisomnográficas

**Features diseñados:**
- Categorías de IMC
- Clasificación de presión arterial
- Flags de obesidad
- Niveles de hipertensión
- Combinaciones de riesgo clínico

### Modelos

Tres algoritmos disponibles:
- **SVM**: Buena precisión con hiperparámetros ajustados
- **Random Forest**: Mejor para análisis de importancia de features
- **Gradient Boosting**: Maneja bien datos desbalanceados

Modos de clasificación:
- **Binario**: Normal vs. Apnea del Sueño (AHI ≥ 5)
- **Multi-clase**: Normal, Leve, Moderada, Severa (umbrales AHI: 5, 15, 30)

### Resultados

El pipeline genera:
- Matrices de confusión
- Reportes de clasificación
- Gráficos de importancia de features
- Curvas ROC y precision-recall
- Gráficos de distribución
- Mapas de calor de correlación

### Fuentes de Datos

- **SHHS**: Estudio de cohorte multi-sitio sobre trastornos del sueño
- **HCHS**: Enfoque en población hispana/latina
- **Fuente**: SleepData.org (National Sleep Research Resource)

### Aviso

Esta es una herramienta de investigación. No usar para diagnóstico médico sin validación clínica. Los modelos entrenados en poblaciones específicas pueden no generalizarse a todas las demografías.

### License

Available for academic and research use. Please cite appropriately in scientific publications.

### Contributing

Open an issue to discuss major changes before submitting a pull request.

### Contact

For questions or collaborations, open an issue in this repository.