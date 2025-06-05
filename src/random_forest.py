"""
Módulo para la construcción y entrenamiento de modelos Random Forest
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score

from scipy.stats import randint
from sklearn.decomposition import PCA
import pandas as pd
from config import *


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import ConfusionMatrixDisplay
        
class RandomForest:
    
    def __init__(self):
        # Parámetros optimizados para Random Forest
        """
        Configuración optimizada para Random Forest basada en mejores prácticas
        """
        
        self.rf_n_estimators = 100
        self.rf_max_depth = None
        self.rf_min_samples_split = 2
        self.rf_min_samples_leaf = 1
        self.rf_max_features = 'sqrt'
        self.rf_bootstrap = True
        self.rf_class_weight = 'balanced'
        
        # Inicializar clasificador con parámetros optimizados
        self.rf_classifier = RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            min_samples_split=self.rf_min_samples_split,
            min_samples_leaf=self.rf_min_samples_leaf,
            max_features=self.rf_max_features,
            bootstrap=self.rf_bootstrap,
            class_weight=self.rf_class_weight,
            random_state=42,
            n_jobs=-1  # Usar todos los núcleos disponibles
        )
        
        # Distribuciones de parámetros para búsqueda aleatoria
        self.param_distributions = {
            'n_estimators': randint(50, 300),
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7],
            'bootstrap': [True, False],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }
                  
    def train_rf_binary(self, X, y):  
        """
        Entrena el modelo de clasificación Random Forest con opciones para manejar desbalance de clases.
        
        Args:
            X: Features
            y: Variable objetivo
            
        Returns:
            Diccionario con resultados del entrenamiento
        """
        print("Entrenando modelo Random Forest...")
        
        # División del dataset con estratificación
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        
        # Entrenamiento del modelo
        self.rf_classifier.fit(X_train, y_train)
        
        # Evaluación
        y_pred = self.rf_classifier.predict(X_test)
        y_prob = self.rf_classifier.predict_proba(X_test)[:, 1]
        
        # Validación cruzada
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.rf_classifier, X, y, cv=cv, scoring='roc_auc'
        )

        # Calcular curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Calcular curva PR
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        return {
            'cv_scores': cv_scores,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'test_data': (X_test, y_test, y_pred, y_prob),
            # 'scaler': scaler,
            'roc_curve': (fpr, tpr),
            'pr_curve': (precision, recall),
            'feature_importance': self.rf_classifier.feature_importances_
        }
    
    def train_rf_multiclase(self, X, y):  
        """
        Entrena el modelo de clasificación Random Forest multiclase con opciones para manejar desbalance de clases.
        
        Args:
            X: Features
            y: Variable objetivo
            
        Returns:
            Diccionario con resultados del entrenamiento
        """
        print("Entrenando modelo Random Forest Multiclase...")
        
        # División del dataset con estratificación
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Entrenamiento del modelo
        self.rf_classifier.fit(X_train, y_train)
        
        # Evaluación
        y_pred = self.rf_classifier.predict(X_test)
        y_prob = self.rf_classifier.predict_proba(X_test)
        
        # Validación cruzada con métrica adecuada para multiclase
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.rf_classifier, X, y, cv=cv, scoring='accuracy'
        )
        
        # Calcular AUC para multiclase
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        
        return {
            'cv_scores': cv_scores,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='macro'),
            'recall': recall_score(y_test, y_pred, average='macro'),
            'f1': f1_score(y_test, y_pred, average='macro'),
            'roc_auc': roc_auc,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'test_data': (X_test, y_test, y_pred, y_prob),
            'feature_importance': self.rf_classifier.feature_importances_
        }
       
    def mostrar_resultados_rf(self, resultados):
        """Muestra los resultados del entrenamiento del Random Forest con más detalle."""
        print("\n========== RESULTADOS DEL ENTRENAMIENTO RANDOM FOREST ==========")
        
        # Puntuaciones CV
        print("\n📊 VALIDACIÓN CRUZADA:")
        cv_scores = resultados['cv_scores']
        print(f"  Puntuaciones: {', '.join([f'{score:.4f}' for score in cv_scores])}")
        print(f"  Media: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # Métricas principales
        print("\n📏 MÉTRICAS EN CONJUNTO DE PRUEBA:")
        if 'accuracy' in resultados:
            print(f"  Accuracy:  {resultados['accuracy']:.4f}")
        if 'precision' in resultados:
            print(f"  Precision: {resultados['precision']:.4f}")
        if 'recall' in resultados:
            print(f"  Recall:    {resultados['recall']:.4f}")
        if 'f1' in resultados:
            print(f"  F1-Score:  {resultados['f1']:.4f}")
        if 'roc_auc' in resultados:
            print(f"  ROC AUC:   {resultados['roc_auc']:.4f}")
        if 'pr_auc' in resultados:
            print(f"  PR AUC:    {resultados['pr_auc']:.4f}")
        
        # Informe de clasificación
        print("\n📋 INFORME DE CLASIFICACIÓN:")
        print(resultados['classification_report'])
        
        # Matriz de confusión
        print("\n🔄 MATRIZ DE CONFUSIÓN:")
        cm = resultados['confusion_matrix']
        print(f"  [ {cm[0][0]}\t{cm[0][1]} ]")
        print(f"  [ {cm[1][0]}\t{cm[1][1]} ]")
        
        # Calcular otras métricas útiles de la matriz de confusión
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        print("\n🔍 MÉTRICAS ADICIONALES:")
        print(f"  Especificidad: {specificity:.4f}")
        print(f"  Valor Predictivo Negativo: {npv:.4f}")
        
        # Importancia de características (específico de Random Forest)
        if 'feature_importance' in resultados:
            print("\n🌳 IMPORTANCIA DE CARACTERÍSTICAS (Top 10):")
            # Nota: necesitarías los nombres de las características para mostrar esto correctamente
            importances = resultados['feature_importance']
            top_indices = importances.argsort()[-10:][::-1]
            for i, idx in enumerate(top_indices):
                print(f"  {i+1}. Característica {idx}: {importances[idx]:.4f}")
    
    def mostrar_resultados_rf_multiclase(self, resultados):
        """Muestra los resultados del entrenamiento del Random Forest multiclase."""
        print("\n========== RESULTADOS DEL ENTRENAMIENTO RANDOM FOREST MULTICLASE ==========")
            
        # Puntuaciones CV
        print("\n📊 VALIDACIÓN CRUZADA:")
        cv_scores = resultados['cv_scores']
        print(f"  Puntuaciones: {', '.join([f'{score:.4f}' for score in cv_scores])}")
        print(f"  Media: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # Métricas principales
        print("\n📏 MÉTRICAS EN CONJUNTO DE PRUEBA:")
        if 'accuracy' in resultados:
            print(f"  Accuracy:  {resultados['accuracy']:.4f}")
        if 'precision' in resultados:
            print(f"  Precision: {resultados['precision']:.4f}")
        if 'recall' in resultados:
            print(f"  Recall:    {resultados['recall']:.4f}")
        if 'f1' in resultados:
            print(f"  F1-Score:  {resultados['f1']:.4f}")
        if 'roc_auc' in resultados:
            print(f"  ROC AUC:   {resultados['roc_auc']:.4f}")
        
        # Informe de clasificación
        print("\n📋 INFORME DE CLASIFICACIÓN:")
        print(resultados['classification_report'])
        
        # Matriz de confusión
        print("\n🔄 MATRIZ DE CONFUSIÓN:")
        cm = resultados['confusion_matrix']
        print(cm)
        
        # Importancia de características (específico de Random Forest)
        if 'feature_importance' in resultados:
            print("\n🌳 IMPORTANCIA DE CARACTERÍSTICAS (Top 10):")
            importances = resultados['feature_importance']
            top_indices = importances.argsort()[-10:][::-1]
            for i, idx in enumerate(top_indices):
                print(f"  {i+1}. Característica {idx}: {importances[idx]:.4f}")
    
    def visualizar_resultados_rf(self, resultados, nombre_modelo='RandomForest', tipo_clasificacion='binaria'):    
        """
        Visualiza los resultados del modelo Random Forest tanto para clasificación binaria como multiclase.
        
        Args:
            resultados: Diccionario con los resultados del entrenamiento
            nombre_modelo: Nombre del modelo para usar en los archivos guardados
            tipo_clasificacion: 'binaria' o 'multiclase' para adaptar las visualizaciones
        """

        # Crear directorio para visualizaciones si no existe
        if not os.path.exists(f'{RF_REPORTS_PATH}'):
            os.makedirs(f'{RF_REPORTS_PATH}')
        
        # Obtener datos de prueba
        X_test_scaled, y_test, y_pred, y_prob = resultados['test_data']
        
        # 1. Métricas del modelo en gráfico de barras
        plt.figure(figsize=(12, 6))
        
        if tipo_clasificacion == 'binaria':
            metricas = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
            valores = [resultados[metrica] for metrica in metricas if metrica in resultados]
            metricas_disponibles = [metrica for metrica in metricas if metrica in resultados]
        else:
            metricas = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            valores = [resultados[metrica] for metrica in metricas]
            metricas_disponibles = metricas
        
        sns.barplot(x=metricas_disponibles, y=valores)
        plt.title(f'Métricas del modelo {nombre_modelo} ({tipo_clasificacion.capitalize()})')
        plt.ylim(0, 1)
        for i, v in enumerate(valores):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{RF_REPORTS_PATH}/metricas_{nombre_modelo}_{tipo_clasificacion}.png')
        plt.close()
        
        # 2. Matriz de confusión
        plt.figure(figsize=(10, 8))
        cm = resultados['confusion_matrix']
        clases = np.unique(y_test)
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                    display_labels=clases)
        disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
        plt.title(f'Matriz de Confusión - {nombre_modelo} ({tipo_clasificacion.capitalize()})')
        plt.tight_layout()
        plt.savefig(f'{RF_REPORTS_PATH}/matriz_confusion_{nombre_modelo}_{tipo_clasificacion}.png')
        plt.close()
        
        # 3. Distribución de probabilidades
        if tipo_clasificacion == 'binaria':
            # Para clasificación binaria: distribución de probabilidades para clase positiva
            plt.figure(figsize=(12, 6))
            
            # Separar probabilidades por clase verdadera
            for clase in np.unique(y_test):
                mask = y_test == clase
                if np.any(mask):
                    sns.kdeplot(y_prob[mask], label=f'Clase real {clase}', alpha=0.7)
            
            plt.title(f'Distribución de Probabilidades - {nombre_modelo}')
            plt.xlabel('Probabilidad (Clase Positiva)')
            plt.ylabel('Densidad')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{RF_REPORTS_PATH}/distribucion_probabilidades_{nombre_modelo}_{tipo_clasificacion}.png')
            plt.close()
            
        else:
            # Para clasificación multiclase: distribución por cada clase
            plt.figure(figsize=(15, 10))
            n_clases = y_prob.shape[1]
            
            for i in range(n_clases):
                plt.subplot(1, n_clases, i+1)
                # Separar probabilidades por clase verdadera
                for j in range(n_clases):
                    mask = y_test == j
                    if np.any(mask):
                        sns.kdeplot(y_prob[mask, i], label=f'Clase real {j}', alpha=0.7)
                
                plt.title(f'Distribución de probabilidad para Clase {i}')
                plt.xlabel('Probabilidad')
                plt.ylabel('Densidad')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'{RF_REPORTS_PATH}/distribucion_probabilidades_{nombre_modelo}_{tipo_clasificacion}.png')
            plt.close()
        
        # 4. Validación cruzada
        plt.figure(figsize=(10, 6))
        cv_scores = resultados['cv_scores']
        plt.boxplot(cv_scores)
        plt.title(f'Distribución de puntuaciones en validación cruzada - {nombre_modelo}')
        plt.ylabel('Score de Validación Cruzada')
        plt.axhline(y=np.mean(cv_scores), color='r', linestyle='--', 
                    label=f'Media: {np.mean(cv_scores):.3f}')
        plt.text(1.1, np.mean(cv_scores), f'{np.mean(cv_scores):.3f}', color='r')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{RF_REPORTS_PATH}/cv_scores_{nombre_modelo}_{tipo_clasificacion}.png')
        plt.close()
        
        # 5. Importancia de características (específico de Random Forest)
        if 'feature_importance' in resultados:
            plt.figure(figsize=(12, 8))
            importances = resultados['feature_importance']
            
            # Obtener índices de las características más importantes
            indices = np.argsort(importances)[::-1]
            top_n = min(15, len(importances))  # Mostrar top 15 o todas si hay menos
            
            # Crear el gráfico de barras
            plt.bar(range(top_n), importances[indices[:top_n]])
            plt.title(f'Importancia de Características - {nombre_modelo}')
            plt.xlabel('Características')
            plt.ylabel('Importancia')
            plt.xticks(range(top_n), [f'Feat_{indices[i]}' for i in range(top_n)], rotation=45)
            plt.tight_layout()
            plt.savefig(f'{RF_REPORTS_PATH}/importancia_caracteristicas_{nombre_modelo}_{tipo_clasificacion}.png')
            plt.close()
            
            # Visualización de características más importantes en scatter plot (solo las top 5)
            # if len(importances) >= 2:
            #     from itertools import combinations
                
            #     # Obtener los índices de las 5 características más importantes
            #     top_features_idx = indices[:min(5, len(importances))]
                
            #     # Crear gráficos de dispersión para pares de características importantes
            #     pairs = list(combinations(top_features_idx, 2))
                
            #     if pairs:
            #         for i, (idx1, idx2) in enumerate(pairs[:3]):  # Limitar a 3 gráficos
            #             plt.figure(figsize=(10, 8))
                        
            #             if tipo_clasificacion == 'binaria':
            #                 scatter = plt.scatter(X_test_scaled[:, idx1], X_test_scaled[:, idx2], 
            #                                     c=y_test, cmap='viridis', alpha=0.7, 
            #                                     edgecolors='w', s=100)
            #             else:
            #                 scatter = plt.scatter(X_test_scaled[:, idx1], X_test_scaled[:, idx2], 
            #                                     c=y_test, cmap='tab10', alpha=0.7, 
            #                                     edgecolors='w', s=100)
                        
            #             # Añadir predicciones incorrectas marcadas con X
            #             mask_incorrect = y_test != y_pred
            #             if np.any(mask_incorrect):
            #                 plt.scatter(X_test_scaled[mask_incorrect, idx1], 
            #                         X_test_scaled[mask_incorrect, idx2],
            #                         marker='x', c='red', s=150, linewidths=2, 
            #                         label='Predicciones incorrectas')
                        
            #             plt.colorbar(scatter, label='Clase')
            #             plt.title(f'Clasificación por características importantes (Feat_{idx1} vs Feat_{idx2})')
            #             plt.xlabel(f'Característica {idx1} (Importancia: {importances[idx1]:.3f})')
            #             plt.ylabel(f'Característica {idx2} (Importancia: {importances[idx2]:.3f})')
            #             if np.any(mask_incorrect):
            #                 plt.legend()
            #             plt.tight_layout()
            #             plt.savefig(f'{RF_REPORTS_PATH}/dispersion_caracteristicas_{idx1}_{idx2}_{nombre_modelo}_{tipo_clasificacion}.png')
            #             plt.close()
        
        # 6. Curvas ROC y PR (solo para clasificación binaria)
        if tipo_clasificacion == 'binaria' and 'roc_curve' in resultados:
            plt.figure(figsize=(15, 5))
            
            # Curva ROC
            plt.subplot(1, 2, 1)
            fpr, tpr = resultados['roc_curve']
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {resultados["roc_auc"]:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Línea aleatoria')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Tasa de Falsos Positivos')
            plt.ylabel('Tasa de Verdaderos Positivos')
            plt.title('Curva ROC')
            plt.legend(loc="lower right")
            
            # Curva Precision-Recall
            if 'pr_curve' in resultados:
                plt.subplot(1, 2, 2)
                precision, recall = resultados['pr_curve']
                plt.plot(recall, precision, color='blue', lw=2,
                        label=f'PR curve (AUC = {resultados["pr_auc"]:.3f})')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Curva Precision-Recall')
                plt.legend(loc="lower left")
            
            plt.tight_layout()
            plt.savefig(f'{RF_REPORTS_PATH}/curvas_roc_pr_{nombre_modelo}_{tipo_clasificacion}.png')
            plt.close()
        
        # 7. Resumen del clasificador
        with open(f'{RF_REPORTS_PATH}/{nombre_modelo}_{tipo_clasificacion}.txt', 'w') as f:
            f.write(f"===== REPORTE DEL MODELO {nombre_modelo} ({tipo_clasificacion.upper()}) =====\n\n")
            f.write(f"Accuracy: {resultados['accuracy']:.4f}\n")
            f.write(f"Precision: {resultados['precision']:.4f}\n")
            f.write(f"Recall: {resultados['recall']:.4f}\n")
            f.write(f"F1 Score: {resultados['f1']:.4f}\n")
            f.write(f"ROC AUC: {resultados['roc_auc']:.4f}\n")
            
            if tipo_clasificacion == 'binaria' and 'pr_auc' in resultados:
                f.write(f"PR AUC: {resultados['pr_auc']:.4f}\n")
            
            f.write("\n===== REPORTE DE CLASIFICACIÓN =====\n\n")
            f.write(resultados['classification_report'])
            f.write("\n\n===== RESULTADOS CV (5-FOLD) =====\n\n")
            f.write(f"Media: {np.mean(resultados['cv_scores']):.4f}\n")
            f.write(f"Desviación estándar: {np.std(resultados['cv_scores']):.4f}\n")
            f.write(f"Min: {np.min(resultados['cv_scores']):.4f}\n")
            f.write(f"Max: {np.max(resultados['cv_scores']):.4f}\n")
            
            # Importancia de características
            if 'feature_importance' in resultados:
                f.write("\n\n===== IMPORTANCIA DE CARACTERÍSTICAS (Top 10) =====\n\n")
                importances = resultados['feature_importance']
                indices = np.argsort(importances)[::-1]
                for i in range(min(10, len(importances))):
                    f.write(f"{i+1}. Característica {indices[i]}: {importances[indices[i]]:.4f}\n")
        
        print(f"Visualizaciones guardadas en el directorio '{RF_REPORTS_PATH}/' para {nombre_modelo} ({tipo_clasificacion})")
        return nombre_modelo