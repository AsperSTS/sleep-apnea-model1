# Sleep Apnea Diagnosis ML Project 🫁

*A machine learning approach to diagnose obstructive and central sleep apnea using harmonized SHHS and HCHS datasets*

## English Version

### 📋 Overview

This project implements machine learning models to diagnose sleep apnea in patients using clinical and demographic data from the harmonized Sleep Heart Health Study (SHHS) and Hispanic Community Health Study (HCHS) datasets available at SleepData.org. The system can predict both binary (presence/absence) and multiclass (severity levels) classifications of sleep apnea.

### 🎯 Objectives

- **Primary Goal**: Develop accurate ML models for sleep apnea diagnosis using non-invasive clinical parameters
- **Secondary Goals**: 
  - Compare performance between different ML algorithms (SVM, Random Forest, Gradient Boosting)
  - Provide interpretable results for clinical decision support
  - Handle class imbalance typical in medical datasets

### 🔬 Scientific Approach

The project follows evidence-based preprocessing and feature engineering techniques:

- **Advanced Imputation**: KNN-based imputation for missing values
- **Clinical Feature Engineering**: BMI categorization, blood pressure classification, risk factor combinations
- **Robust Outlier Detection**: Conservative 3-IQR method to preserve medical data integrity
- **Class Balancing**: SMOTE/ADASYN techniques for handling imbalanced datasets
- **Feature Selection**: Hybrid approach using mutual information and F-score methods

### 🏗️ Project Structure

```
sleep-apnea-diagnosis/
├──test
├──src
│   ├── train.py             # Main training script
│   ├── preprocessing.py     # Data preprocessing and feature engineering
│   ├── eda.py               # Exploratory Data Analysis
│   ├── prepare_data.py      # Data preparation for ML models
│   ├── predict.py           # Prediction script for new patients
│   ├── config.py            # Configuration parameters
│   ├── utils.py             # Useful functions
│   ├── svm.py               # Svm implementation               
│   ├── random_forest.py     # Random Forest implementation
│   ├── gradient_boost.py    # Gradient Boost implementation
│   ├── models/              # Store the models
│   ├── reports/             # Store the reports and the graphics
│   ├── train_test/          # Store the Test and Train datasets
│   ├── visual_eda/          # Store the Exploratory Data Analysis graphics
│   ├── visual_pre/          # Store the Preprocessing graphics and analytics  
│   └── visual_model/        

```

### 🚀 Quick Start

#### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn missingno
```

#### Basic Usage

1. **Complete Analysis with Visualizations (no training)**:
   ```bash
   python train.py --eda
   ```

2. **Training with Binary Classification**:
   ```bash
   python train.py --train --modo binario
   ```

3. **Training with Multiclass Severity Classification**:
   ```bash
   python train.py --train --modo multiclase
   ```


### 📊 Features

**Clinical Parameters Used:**
- Age, BMI, Blood Pressure (systolic/diastolic)
- Smoking history (current/ever smoker)
- Gender, Race/Ethnicity
- Arousal index and other polysomnographic measures

**Engineered Features:**
- BMI categories (WHO classification)
- Blood pressure classification (AHA guidelines)
- Obesity indicators (BMI ≥30, ≥35)
- Hypertension severity levels
- Combined clinical risk scores

### 🎯 Model Performance

The system supports three main algorithms:
- **Support Vector Machine (SVM)**: High accuracy with proper hyperparameter tuning
- **Random Forest**: Excellent feature importance interpretation
- **Gradient Boosting**: Strong performance on imbalanced datasets

Classification modes:
- **Binary**: Normal vs. Sleep Apnea (AHI ≥ 5)
- **Multiclass**: Normal, Mild, Moderate, Severe (based on AHI thresholds: 5, 15, 30)

### 📈 Results and Visualization

The system automatically generates:
- Confusion matrices and classification reports
- Feature importance plots
- ROC curves and precision-recall curves
- Distribution analysis of clinical variables
- Correlation heatmaps

### 🔍 Data Sources

This project uses harmonized datasets from:
- **SHHS (Sleep Heart Health Study)**: Multi-site cohort study of sleep disorders
- **HCHS (Hispanic Community Health Study)**: Focus on Hispanic/Latino populations
- **Source**: SleepData.org - National Sleep Research Resource

### ⚠️ Important Notes

- This is a research/educational tool and should not replace professional medical diagnosis
- Results should be validated with clinical expertise
- The model is trained on specific populations and may not generalize to all demographics

---

## Versión en Español

### 📋 Descripción General

Este proyecto implementa modelos de aprendizaje automático para diagnosticar apnea del sueño en pacientes utilizando datos clínicos y demográficos de las bases de datos armonizadas SHHS y HCHS disponibles en SleepData.org. El sistema puede predecir tanto clasificaciones binarias (presencia/ausencia) como multiclase (niveles de severidad) de apnea del sueño.

### 🎯 Objetivos

- **Objetivo Principal**: Desarrollar modelos de ML precisos para el diagnóstico de apnea del sueño usando parámetros clínicos no invasivos
- **Objetivos Secundarios**:
  - Comparar el rendimiento entre diferentes algoritmos de ML (SVM, Random Forest, Gradient Boosting)
  - Proporcionar resultados interpretables para apoyo en decisiones clínicas
  - Manejar el desbalance de clases típico en conjuntos de datos médicos

### 🔬 Enfoque Científico

El proyecto sigue técnicas de preprocesamiento e ingeniería de características basadas en evidencia:

- **Imputación Avanzada**: Imputación basada en KNN para valores faltantes
- **Ingeniería de Características Clínicas**: Categorización de IMC, clasificación de presión arterial, combinaciones de factores de riesgo
- **Detección Robusta de Valores Atípicos**: Método conservador 3-IQR para preservar la integridad de los datos médicos
- **Balanceo de Clases**: Técnicas SMOTE/ADASYN para manejar conjuntos de datos desbalanceados
- **Selección de Características**: Enfoque híbrido usando información mutua y métodos F-score

### 🚀 Inicio Rápido

#### Uso Básico

1. **Análisis Completo con Visualizaciones (sin entrenamiento)**:
   ```bash
   python train.py --eda
   ```

2. **Entrenamiento con Clasificación Binaria**:
   ```bash
   python train.py --train --modo binario
   ```

3. **Entrenamiento con Clasificación Multiclase de Severidad**:
   ```bash
   python train.py --train --modo multiclase
   ```


### 📊 Características

**Parámetros Clínicos Utilizados:**
- Edad, IMC, Presión Arterial (sistólica/diastólica)
- Historial de tabaquismo (fumador actual/alguna vez)
- Género, Raza/Etnia
- Índice de despertar y otras medidas polisomnográficas

**Características Diseñadas:**
- Categorías de IMC (clasificación OMS)
- Clasificación de presión arterial (guías AHA)
- Indicadores de obesidad (IMC ≥30, ≥35)
- Niveles de severidad de hipertensión
- Puntuaciones combinadas de riesgo clínico

### 🎯 Rendimiento del Modelo

El sistema soporta tres algoritmos principales:
- **Support Vector Machine (SVM)**: Alta precisión con ajuste adecuado de hiperparámetros
- **Random Forest**: Excelente interpretación de importancia de características
- **Gradient Boosting**: Fuerte rendimiento en conjuntos de datos desbalanceados

Modos de clasificación:
- **Binario**: Normal vs. Apnea del Sueño (AHI ≥ 5)
- **Multiclase**: Normal, Leve, Moderada, Severa (basado en umbrales AHI: 5, 15, 30)

### ⚠️ Notas Importantes

- Esta es una herramienta de investigación/educación y no debe reemplazar el diagnóstico médico profesional
- Los resultados deben ser validados con experiencia clínica
- El modelo está entrenado en poblaciones específicas y puede no generalizarse a todas las demografías

### 🔍 Fuentes de Datos

Este proyecto utiliza conjuntos de datos armonizados de:
- **SHHS (Sleep Heart Health Study)**: Estudio de cohorte multi-sitio de trastornos del sueño
- **HCHS (Hispanic Community Health Study)**: Enfoque en poblaciones hispanas/latinas
- **Fuente**: SleepData.org - Recurso Nacional de Investigación del Sueño

### 📝 Licencia

Este proyecto está disponible para uso académico y de investigación. Por favor, cite apropiadamente si utiliza este código en publicaciones científicas.

### 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abra un issue para discutir cambios mayores antes de enviar un pull request.

### 📞 Contacto

Para preguntas sobre el proyecto o colaboraciones, por favor abra un issue en este repositorio.