"""
Módulo para la construcción y entrenamiento de modelos
"""
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, StratifiedKFold
# Calcular métricas adicionales
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
        
from scipy.stats import uniform, randint
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from config import *
from itertools import combinations


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
from sklearn.metrics import ConfusionMatrixDisplay
        
class SVM:
    def __init__(self):
        # Expanded parameter space
        """
            Score: 0.8061 (±0.0150)
            C: 10.496128632644757
            class_weight: None
            degree: 2
            gamma: 2.9123914019804196
            kernel: poly
            tol: 0.01
        """
        
        self.svm_c_parameter = 10.496128632644757  
        self.svm_kernel_parameter = 'poly'  
        self.svm_gamma_parameter = 2.9123914019804196 
        self.svm_tolerance_parameter = 0.01
        self.svm_class_weight_parameter = None # Added for class imbalance
        # Initialize classifier with optimized defaults
        self.svm_classifier = SVC(
            kernel=self.svm_kernel_parameter, 
            probability=True, 
            C=self.svm_c_parameter, 
            tol=self.svm_tolerance_parameter, 
            class_weight=self.svm_class_weight_parameter, 
            gamma=self.svm_gamma_parameter, 
            degree=2,
            cache_size=1000  # Added for speed improvement
        )
        self.svm_precision_result = None
        
        # Expanded parameter distributions
        self.param_distributions = {
            'C': uniform(0.001, 30.0),  # Wider range
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': uniform(0.0001, 20.0),  # Wider range
            'class_weight': ['balanced', None],
            'degree': randint(2, 6),  # Expanded range for poly kernel
            'tol': [0.0001, 0.001, 0.01]  # Added tolerance exploration
        }  
                       
    def train_svm_binary(self, X, y):  
        """
        Entrena el modelo de clasificación SVM con opciones para manejar desbalance de clases.
        
        Args:
            X: Features
            y: Variable objetivo
            
        Returns:
            Diccionario con resultados del entrenamiento
        """
        print("Entrenando modelo SVM...")
        
        # División del dataset con estratificación
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Entrenamiento del modelo
        self.svm_classifier.fit(X_train, y_train)
        
        # Evaluación
        y_pred = self.svm_classifier.predict(X_test)
        y_prob = self.svm_classifier.predict_proba(X_test)[:, 1] # Dos clases
        
        # Validación cruzada
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.svm_classifier, X, y, cv=cv, scoring='roc_auc'
        ) # Dos clases

        # Métricas avanzadas

        # Calcular curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr) # Dos clases
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
            # 'scaler': scaler,  # Guardar para aplicar a nuevos datos
            'roc_curve': (fpr, tpr),
            'pr_curve': (precision, recall)
        }
    
    def train_svm_multiclase(self, X, y):  
        """
        Entrena el modelo de clasificación SVM con opciones para manejar desbalance de clases.
        
        Args:
            X: Features
            y: Variable objetivo
            
        Returns:
            Diccionario con resultados del entrenamiento
        """
        print("Entrenando modelo SVM Multiclase...")
        
        # División del dataset con estratificación
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Entrenamiento del modelo
        self.svm_classifier.fit(X_train, y_train)
        
        # Evaluación
        y_pred = self.svm_classifier.predict(X_test)
        y_prob = self.svm_classifier.predict_proba(X_test)
        
        # Validación cruzada con métrica adecuada para multiclase
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.svm_classifier, X, y, cv=cv, scoring='accuracy'  # Cambia a accuracy o f1_macro
        )
        
        # Métricas para multiclase
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
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
            # Elimina 'roc_curve' y 'pr_curve' que son para binarios
        } 
       
    def mostrar_resultados_svm(self, resultados):
        """Muestra los resultados del entrenamiento del SVM con más detalle."""
        print("\n========== RESULTADOS DEL ENTRENAMIENTO SVM ==========")
        
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
    
    def mostrar_resultados_svm_multiclase(self, resultados):
        print("\n========== RESULTADOS DEL ENTRENAMIENTO SVM ==========")
            
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
        # Imprimir matriz de confusión para multiclase
        print(cm)
    
    def mostrar_resultados_randomized_search_binary(self, resultados, nombre_archivo = "svm_randomizer_search_results.txt"):
        """
        Muestra los resultados mejorados de RandomizedSearchCV y los guarda en un archivo de texto.

        Args:
            resultados (dict): Un diccionario que contiene los resultados de RandomizedSearchCV,
                            incluyendo 'best_params', 'best_score', 'cv_results',
                            opcionalmente 'roc_auc', 'pr_auc', 'classification_report',
                            y 'confusion_matrix'.
            nombre_archivo (str): El nombre del archivo .txt donde se guardarán los resultados.
        """
        
        # Abrir el archivo en modo escritura. 'w' creará el archivo o lo sobrescribirá si ya existe.
        # 'encoding="utf-8"' es importante para manejar caracteres especiales.
        with open(nombre_archivo, 'w', encoding='utf-8') as f:
            # Redirigir la salida estándar (stdout) al archivo
            sys.stdout = f

            # --- Contenido a imprimir en el archivo ---
            print("Muestra los resultados mejorados de RandomizedSearchCV.")
            print("\n========== RESULTADOS DE RANDOMIZED SEARCH CV ==========")
            
            # Mejores parámetros
            print("\n🏆 MEJORES PARÁMETROS:")
            for param, value in resultados['best_params'].items():
                print(f"  {param}: {value}")
            
            print(f"\n📈 MEJOR PUNTUACIÓN: {resultados['best_score']:.4f}")
            
            # Top 5 combinaciones
            print("\n🔝 TOP 5 COMBINACIONES DE PARÁMETROS:")
            cv_results = resultados['cv_results']
            # Obtener índices ordenados por mean_test_score (descendente)
            sorted_indices = cv_results['mean_test_score'].argsort()[::-1][:5]
            
            for i, idx in enumerate(sorted_indices):
                mean = cv_results['mean_test_score'][idx]
                std = cv_results['std_test_score'][idx]
                params = cv_results['params'][idx]
                print(f"  {i+1}. Score: {mean:.4f} (±{std:.4f})")
                for param, value in params.items():
                    print(f"     {param}: {value}")
                print("")
            
            # Métricas en test
            if 'roc_auc' in resultados:
                print(f"\n📊 ROC AUC en test: {resultados['roc_auc']:.4f}")
            if 'pr_auc' in resultados:
                print(f"📊 PR AUC en test: {resultados['pr_auc']:.4f}")
            
            # Informe de clasificación
            print("\n📋 INFORME DE CLASIFICACIÓN:")
            print(resultados['classification_report'])
            
            # Matriz de confusión
            print("\n🔄 MATRIZ DE CONFUSIÓN:")
            cm = resultados['confusion_matrix']
            print(f"  [ {cm[0][0]}\t{cm[0][1]} ]")
            print(f"  [ {cm[1][0]}\t{cm[1][1]} ]")
            # --- Fin del contenido ---

        # Restaurar la salida estándar a la consola
        sys.stdout = sys.__stdout__
        print(f"Resultados guardados exitosamente en '{nombre_archivo}'")
        
    def mostrar_resultados_randomized_search_multiclase(self, resultados):
        """
        Muestra los resultados mejorados de RandomizedSearchCV para problemas multiclase.
        
        Args:
            resultados: Diccionario con resultados del proceso de búsqueda de parámetros
        """
        print("\n========== RESULTADOS DE RANDOMIZED SEARCH CV MULTICLASE ==========")
        
        # Mejores parámetros
        print("\n🏆 MEJORES PARÁMETROS:")
        for param, value in resultados['best_params'].items():
            print(f"  {param}: {value}")
        
        print(f"\n📈 MEJOR PUNTUACIÓN: {resultados['best_score']:.4f}")
        
        # Top 5 combinaciones
        print("\n🔝 TOP 5 COMBINACIONES DE PARÁMETROS:")
        cv_results = resultados['cv_results']
        # Obtener índices ordenados por mean_test_score (descendente)
        sorted_indices = cv_results['mean_test_score'].argsort()[::-1][:5]
        
        for i, idx in enumerate(sorted_indices):
            mean = cv_results['mean_test_score'][idx]
            std = cv_results['std_test_score'][idx]
            params = cv_results['params'][idx]
            print(f"  {i+1}. Score: {mean:.4f} (±{std:.4f})")
            for param, value in params.items():
                print(f"     {param}: {value}")
            print("")
        
        # Métricas en test para multiclase
        print("\n📊 MÉTRICAS DE EVALUACIÓN EN TEST:")
        if 'accuracy' in resultados:
            print(f"  Accuracy: {resultados['accuracy']:.4f}")
        if 'precision' in resultados:
            print(f"  Precision (macro): {resultados['precision']:.4f}")
        if 'recall' in resultados:
            print(f"  Recall (macro): {resultados['recall']:.4f}")
        if 'f1' in resultados:
            print(f"  F1 Score (macro): {resultados['f1']:.4f}")
        if 'roc_auc' in resultados and resultados['roc_auc'] is not None:
            print(f"  ROC AUC (ovr): {resultados['roc_auc']:.4f}")
        
        # Informe de clasificación
        print("\n📋 INFORME DE CLASIFICACIÓN:")
        print(resultados['classification_report'])
        
        # Matriz de confusión
        print("\n🔄 MATRIZ DE CONFUSIÓN:")
        cm = resultados['confusion_matrix']
        
        # Determinar el número de clases a partir de la matriz
        n_classes = len(cm)
        
        # Mostrar la matriz de confusión de forma más legible
        print("  ", end="")
        for i in range(n_classes):
            print(f"[{i}]", end="\t")
        print("")
        
        for i in range(n_classes):
            print(f"  [{i}]", end=" ")
            for j in range(n_classes):
                print(f"{cm[i][j]}", end="\t")
            print("")
        
        # Opcionalmente, podemos agregar una interpretación más detallada
        print("\n📉 ANÁLISIS DE LA MATRIZ DE CONFUSIÓN:")
        for i in range(n_classes):
            total_true = sum(cm[i])
            if total_true > 0:
                correct = cm[i][i]
                print(f"  Clase {i}: {correct}/{total_true} correctamente clasificados ({correct/total_true*100:.1f}%)")
                
                # Identificar principales confusiones
                if n_classes > 2:  # Solo relevante para más de 2 clases
                    confusions = [(j, cm[i][j]) for j in range(n_classes) if j != i and cm[i][j] > 0]
                    if confusions:
                        confusions.sort(key=lambda x: x[1], reverse=True)
                        print(f"    Principales confusiones: ", end="")
                        for cls, count in confusions[:2]:  # Top 2 confusiones
                            print(f"con clase {cls} ({count} casos, {count/total_true*100:.1f}%)", end="; ")
                        print("")
    
    def visualizar_resultados_binario(self, resultados, nombre_modelo='SVM_Binario'):    
        """
        Visualiza los resultados del modelo SVM binario.
        
        Args:
            resultados: Diccionario con los resultados del entrenamiento
            nombre_modelo: Nombre del modelo para usar en los archivos guardados
        """

        
        # Crear directorio para visualizaciones si no existe
        if not os.path.exists(f'{VISUAL_MODEL_DIR}'):
            os.makedirs(f'{VISUAL_MODEL_DIR}')
        
        # Obtener datos de prueba
        X_test_scaled, y_test, y_pred, y_prob = resultados['test_data']
        
        # 1. Métricas del modelo en gráfico de barras
        plt.figure(figsize=(12, 6))
        metricas = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        valores = [resultados[metrica] for metrica in metricas]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        bars = plt.bar(metricas, valores, color=colors)
        plt.title(f'Métricas del modelo {nombre_modelo}', fontsize=14, fontweight='bold')
        plt.ylim(0, 1)
        plt.ylabel('Puntuación', fontsize=12)
        
        # Añadir valores en las barras
        for i, (bar, v) in enumerate(zip(bars, valores)):
            plt.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{VISUAL_MODEL_DIR}/metricas_{nombre_modelo}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Matriz de confusión
        plt.figure(figsize=(8, 6))
        cm = resultados['confusion_matrix']
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                    display_labels=['No Apnea', 'Apnea'])
        disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
        plt.title(f'Matriz de Confusión - {nombre_modelo}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{VISUAL_MODEL_DIR}/matriz_confusion_{nombre_modelo}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Curva ROC
        plt.figure(figsize=(8, 6))
        fpr, tpr = resultados['roc_curve']
        roc_auc = resultados['roc_auc']
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'Curva ROC (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Línea aleatoria')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title(f'Curva ROC - {nombre_modelo}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{VISUAL_MODEL_DIR}/curva_roc_{nombre_modelo}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Curva Precision-Recall
        plt.figure(figsize=(8, 6))
        precision, recall = resultados['pr_curve']
        pr_auc = resultados['pr_auc']
        
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'Curva PR (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Curva Precision-Recall - {nombre_modelo}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{VISUAL_MODEL_DIR}/curva_pr_{nombre_modelo}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Distribución de probabilidades
        plt.figure(figsize=(10, 6))
        
        # Separar probabilidades por clase real
        prob_clase_0 = y_prob[y_test == 0]
        prob_clase_1 = y_prob[y_test == 1]
        
        plt.hist(prob_clase_0, bins=30, alpha=0.7, label='No Apnea (Clase 0)', 
                color='lightblue', density=True)
        plt.hist(prob_clase_1, bins=30, alpha=0.7, label='Apnea (Clase 1)', 
                color='lightcoral', density=True)
        
        plt.xlabel('Probabilidad predicha para Clase 1')
        plt.ylabel('Densidad')
        plt.title(f'Distribución de Probabilidades Predichas - {nombre_modelo}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{VISUAL_MODEL_DIR}/distribucion_probabilidades_{nombre_modelo}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Validación cruzada
        plt.figure(figsize=(8, 6))
        cv_scores = resultados['cv_scores']
        
        plt.boxplot(cv_scores, patch_artist=True, 
                    boxprops=dict(facecolor='lightblue', alpha=0.7))
        plt.title(f'Distribución de puntuaciones en validación cruzada - {nombre_modelo}')
        plt.ylabel('ROC AUC Score')
        plt.axhline(y=np.mean(cv_scores), color='r', linestyle='--', 
                    label=f'Media: {np.mean(cv_scores):.3f}')
        plt.text(1.1, np.mean(cv_scores), f'{np.mean(cv_scores):.3f}', 
                color='r', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{VISUAL_MODEL_DIR}/cv_scores_{nombre_modelo}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # # 7. Visualización de clasificación por pares de características (si hay coeficientes)
        # if hasattr(self.svm_classifier, 'coef_') and self.svm_classifier.coef_ is not None:
        #     from itertools import combinations
            
        #     # Obtener las características más importantes según los coeficientes
        #     feature_importance = np.abs(self.svm_classifier.coef_[0])
            
        #     # Obtener los índices de las características más importantes
        #     top_features_idx = np.argsort(feature_importance)[-5:] if len(feature_importance) > 5 else np.argsort(feature_importance)
            
        #     # Crear gráficos de dispersión para pares de características importantes
        #     pairs = list(combinations(top_features_idx, 2))
            
        #     if pairs:
        #         for i, (idx1, idx2) in enumerate(pairs[:3]):  # Limitar a 3 gráficos
        #             plt.figure(figsize=(10, 8))
                    
        #             scatter = plt.scatter(X_test_scaled[:, idx1], X_test_scaled[:, idx2], 
        #                                 c=y_test, cmap='RdYlBu', alpha=0.7, 
        #                                 edgecolors='w', s=60)
                    
        #             # Añadir predicciones incorrectas marcadas con X
        #             mask_incorrect = y_test != y_pred
        #             if np.any(mask_incorrect):
        #                 plt.scatter(X_test_scaled[mask_incorrect, idx1], 
        #                         X_test_scaled[mask_incorrect, idx2],
        #                         marker='x', c='red', s=150, linewidths=2, 
        #                         label='Predicciones incorrectas')
                    
        #             plt.colorbar(scatter, label='Clase')
        #             plt.title(f'Clasificación en características importantes ({idx1} vs {idx2})')
        #             plt.xlabel(f'Característica {idx1}')
        #             plt.ylabel(f'Característica {idx2}')
        #             if np.any(mask_incorrect):
        #                 plt.legend()
        #             plt.grid(True, alpha=0.3)
        #             plt.tight_layout()
        #             plt.savefig(f'{VISUAL_MODEL_DIR}/dispersion_caracteristicas_{idx1}_{idx2}_{nombre_modelo}.png', 
        #                     dpi=300, bbox_inches='tight')
        #             plt.close()
        
        # 8. Resumen del clasificador
        with open(f'{SVM_REPORTS_PATH}/reporte_{nombre_modelo}.txt', 'w', encoding='utf-8') as f:
            f.write(f"===== REPORTE DEL MODELO {nombre_modelo} =====\n\n")
            f.write(f"Accuracy: {resultados['accuracy']:.4f}\n")
            f.write(f"Precision: {resultados['precision']:.4f}\n")
            f.write(f"Recall: {resultados['recall']:.4f}\n")
            f.write(f"F1 Score: {resultados['f1']:.4f}\n")
            f.write(f"ROC AUC: {resultados['roc_auc']:.4f}\n")
            f.write(f"PR AUC: {resultados['pr_auc']:.4f}\n\n")
            f.write("===== REPORTE DE CLASIFICACIÓN =====\n\n")
            f.write(resultados['classification_report'])
            f.write("\n\n===== RESULTADOS CV (5-FOLD) =====\n\n")
            f.write(f"Media: {np.mean(resultados['cv_scores']):.4f}\n")
            f.write(f"Desviación estándar: {np.std(resultados['cv_scores']):.4f}\n")
            f.write(f"Min: {np.min(resultados['cv_scores']):.4f}\n")
            f.write(f"Max: {np.max(resultados['cv_scores']):.4f}\n")
            
            # Información adicional de la matriz de confusión
            cm = resultados['confusion_matrix']
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            f.write(f"\n===== MÉTRICAS ADICIONALES =====\n\n")
            f.write(f"Especificidad: {specificity:.4f}\n")
            f.write(f"Valor Predictivo Negativo: {npv:.4f}\n")
            f.write(f"Verdaderos Negativos: {tn}\n")
            f.write(f"Falsos Positivos: {fp}\n")
            f.write(f"Falsos Negativos: {fn}\n")
            f.write(f"Verdaderos Positivos: {tp}\n")
        
        # print(f"Visualizaciones guardadas en el directorio '{VISUAL_MODEL_DIR}/'")
        return nombre_modelo         
        
    def visualizar_resultados_multiclase(self, resultados, nombre_modelo='SVM_multiclase'):    
    
        """
        Visualiza los resultados del modelo SVM multiclase.
        
        Args:
            resultados: Diccionario con los resultados del entrenamiento
            nombre_modelo: Nombre del modelo para usar en los archivos guardados
        """

        # Crear directorio para visualizaciones si no existe
        if not os.path.exists('{VISUAL_MODEL_DIR}'):
            os.makedirs('{VISUAL_MODEL_DIR}')
        
        # Obtener datos de prueba
        X_test_scaled, y_test, y_pred, y_prob = resultados['test_data']
        
        # 1. Métricas del modelo en gráfico de barras
        plt.figure(figsize=(12, 6))
        metricas = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        valores = [resultados[metrica] for metrica in metricas]
        
        sns.barplot(x=metricas, y=valores)
        plt.title(f'Métricas del modelo {nombre_modelo}')
        plt.ylim(0, 1)
        for i, v in enumerate(valores):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
        plt.savefig(f'{VISUAL_MODEL_DIR}/metricas_{nombre_modelo}.png')
        plt.close()
        
        # 2. Matriz de confusión
        plt.figure(figsize=(10, 8))
        cm = resultados['confusion_matrix']
        clases = np.unique(y_test)
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                    display_labels=clases)
        disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
        plt.title(f'Matriz de Confusión - {nombre_modelo}')
        plt.savefig(f'{VISUAL_MODEL_DIR}/matriz_confusion_{nombre_modelo}.png')
        plt.close()
        
        # 3. Distribución de probabilidades por clase
        plt.figure(figsize=(15, 10))
        n_clases = y_prob.shape[1]
        
        for i in range(n_clases):
            plt.subplot(1, n_clases, i+1)
            # Separar probabilidades por clase verdadera
            for j in range(n_clases):
                mask = y_test == j
                if np.any(mask):
                    sns.kdeplot(y_prob[mask, i], label=f'Clase real {j}')
            
            plt.title(f'Distribución de probabilidad para Clase {i}')
            plt.xlabel('Probabilidad')
            plt.ylabel('Densidad')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{VISUAL_MODEL_DIR}/distribucion_probabilidades_{nombre_modelo}.png')
        plt.close()
        
        # 4. Validación cruzada
        plt.figure(figsize=(10, 6))
        cv_scores = resultados['cv_scores']
        plt.boxplot(cv_scores)
        plt.title(f'Distribución de puntuaciones en validación cruzada - {nombre_modelo}')
        plt.ylabel('Accuracy')
        plt.axhline(y=np.mean(cv_scores), color='r', linestyle='--', 
                    label=f'Media: {np.mean(cv_scores):.3f}')
        plt.text(1.1, np.mean(cv_scores), f'{np.mean(cv_scores):.3f}', color='r')
        plt.savefig(f'{VISUAL_MODEL_DIR}/cv_scores_{nombre_modelo}.png')
        plt.close()
        
        # # 5. Visualización de clasificación por pares de características (si hay coeficientes)
        # if hasattr(self.svm_classifier, 'coef_'):
            
            
        #     # Obtener las características más importantes según los coeficientes
        #     # Suma la importancia absoluta de cada característica a través de todas las clases
        #     if self.svm_classifier.coef_.ndim > 1:
        #         feature_importance = np.sum(np.abs(self.svm_classifier.coef_), axis=0)
        #     else:
        #         feature_importance = np.abs(self.svm_classifier.coef_)
            
        #     # Obtener los índices de las características más importantes
        #     top_features_idx = np.argsort(feature_importance)[-5:] if len(feature_importance) > 5 else np.argsort(feature_importance)
            
        #     # Crear gráficos de dispersión para pares de características importantes
        #     pairs = list(combinations(top_features_idx, 2))
            
        #     if pairs:
        #         for i, (idx1, idx2) in enumerate(pairs[:3]):  # Limitar a 3 gráficos
        #             plt.figure(figsize=(10, 8))
                    
        #             scatter = plt.scatter(X_test_scaled[:, idx1], X_test_scaled[:, idx2], 
        #                                 c=y_test, cmap='viridis', alpha=0.7, 
        #                                 edgecolors='w', s=100)
                    
        #             # Añadir predicciones incorrectas marcadas con X
        #             mask_incorrect = y_test != y_pred
        #             if np.any(mask_incorrect):
        #                 plt.scatter(X_test_scaled[mask_incorrect, idx1], 
        #                         X_test_scaled[mask_incorrect, idx2],
        #                         marker='x', c='red', s=150, linewidths=2, 
        #                         label='Predicciones incorrectas')
                    
        #             plt.colorbar(scatter, label='Clase')
        #             plt.title(f'Clasificación en características importantes ({idx1} vs {idx2})')
        #             plt.xlabel(f'Característica {idx1}')
        #             plt.ylabel(f'Característica {idx2}')
        #             plt.legend()
        #             plt.savefig(f'{VISUAL_MODEL_DIR}/dispersion_caracteristicas_{idx1}_{idx2}_{nombre_modelo}.png')
        #             plt.close()
        
        # 6. Resumen del clasificador
        with open(f'{SVM_REPORTS_PATH}/reporte_{nombre_modelo}.txt', 'w') as f:
            f.write(f"===== REPORTE DEL MODELO {nombre_modelo} =====\n\n")
            f.write(f"Accuracy: {resultados['accuracy']:.4f}\n")
            f.write(f"Precision: {resultados['precision']:.4f}\n")
            f.write(f"Recall: {resultados['recall']:.4f}\n")
            f.write(f"F1 Score: {resultados['f1']:.4f}\n")
            f.write(f"ROC AUC: {resultados['roc_auc']:.4f}\n\n")
            f.write("===== REPORTE DE CLASIFICACIÓN =====\n\n")
            f.write(resultados['classification_report'])
            f.write("\n\n===== RESULTADOS CV (5-FOLD) =====\n\n")
            f.write(f"Media: {np.mean(resultados['cv_scores']):.4f}\n")
            f.write(f"Desviación estándar: {np.std(resultados['cv_scores']):.4f}\n")
            f.write(f"Min: {np.min(resultados['cv_scores']):.4f}\n")
            f.write(f"Max: {np.max(resultados['cv_scores']):.4f}\n")
        
        print(f"Visualizaciones guardadas en el directorio '{VISUAL_MODEL_DIR}/'")
        return nombre_modelo
    
    def find_best_parameters_binary(self, X, y, n_iter=50, cv_folds=5, scoring='recall'):
        """
        Búsqueda optimizada de hiperparámetros con RandomizedSearchCV.
        
        Args:
            X: Features
            y: Variable objetivo
            n_iter: Número de combinaciones a probar
            cv_folds: Número de particiones para validación cruzada
            scoring: Métrica para optimizar ('roc_auc', 'f1', 'precision', 'recall')
            
        Returns:
            Diccionario con resultados de la búsqueda
        """
        print(f"Optimizando hiperparámetros con {n_iter} iteraciones...")
        
        # División del dataset con estratificación
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # # Estandarizar características
        # scaler = StandardScaler()
        # X_train_scaled = scaler.fit_transform(X_train)
        # X_test_scaled = scaler.transform(X_test)
        
        # Configurar y ejecutar RandomizedSearchCV
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        random_search = RandomizedSearchCV(
            estimator=self.svm_classifier,
            param_distributions=self.param_distributions,
            n_iter=n_iter,
            cv=cv,
            verbose=2,
            scoring=scoring,
            random_state=42,
            n_jobs=-1,
            return_train_score=True
        )
        
        # Entrenar el modelo
        random_search.fit(X_train, y_train)
        
        # Actualizar el clasificador con los mejores parámetros
        self.svm_classifier = random_search.best_estimator_
        
        # Guardar los mejores parámetros en los atributos de la clase
        best_params = random_search.best_params_
        if 'C' in best_params:
            self.svm_c_parameter = best_params['C']
        if 'kernel' in best_params:
            self.svm_kernel_parameter = best_params['kernel']
        if 'gamma' in best_params:
            self.svm_gamma_parameter = best_params['gamma']
        if 'class_weight' in best_params:
            self.svm_class_weight_parameter = best_params['class_weight']
        
        # Evaluación en conjunto de prueba
        y_pred = self.svm_classifier.predict(X_test)
        y_prob = self.svm_classifier.predict_proba(X_test)[:, 1]
        
        
        # Curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Curva PR
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        # Retornar resultados completos
        return {
            'cv_results': random_search.cv_results_,
            'best_params': best_params,
            'best_score': random_search.best_score_,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'test_data': (X_test, y_test, y_pred, y_prob),
            # 'scaler': scaler,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'roc_curve': (fpr, tpr),
            'pr_curve': (precision, recall)
        }

    def find_best_parameters_multiclass(self, X, y, n_iter=40, cv_folds=5, scoring='recall_macro'):
        """
        Búsqueda optimizada de hiperparámetros con RandomizedSearchCV para problemas multiclase.
        
        Args:
            X: Features
            y: Variable objetivo (multiclase)
            n_iter: Número de combinaciones a probar
            cv_folds: Número de particiones para validación cruzada
            scoring: Métrica para optimizar ('accuracy', 'f1_macro', 'precision_macro', 'recall_macro')
            
        Returns:
            Diccionario con resultados de la búsqueda
        """
        print(f"Optimizando hiperparámetros para clasificación multiclase con {n_iter} iteraciones...")
        
        # División del dataset con estratificación
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Estandarizar características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Configurar y ejecutar RandomizedSearchCV
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        random_search = RandomizedSearchCV(
            estimator=self.svm_classifier,
            param_distributions=self.param_distributions,
            n_iter=n_iter,
            cv=cv,
            verbose=2,
            scoring=scoring,  # Para multiclase: 'accuracy', 'f1_macro', 'precision_macro', 'recall_macro'
            random_state=42,
            n_jobs=-1,
            return_train_score=True
        )
        
        # Entrenar el modelo
        random_search.fit(X_train_scaled, y_train)
        
        # Actualizar el clasificador con los mejores parámetros
        self.svm_classifier = random_search.best_estimator_
        
        # Guardar los mejores parámetros en los atributos de la clase
        best_params = random_search.best_params_
        if 'C' in best_params:
            self.svm_c_parameter = best_params['C']
        if 'kernel' in best_params:
            self.svm_kernel_parameter = best_params['kernel']
        if 'gamma' in best_params:
            self.svm_gamma_parameter = best_params['gamma']
        if 'class_weight' in best_params:
            self.svm_class_weight_parameter = best_params['class_weight']
        
        # Evaluación en conjunto de prueba
        y_pred = self.svm_classifier.predict(X_test_scaled)
        y_prob = self.svm_classifier.predict_proba(X_test_scaled)
        
        # Calcular métricas para multiclase
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, 
            roc_auc_score, classification_report, confusion_matrix
        )
        
        # Calcular métricas específicas para multiclase
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        
        # ROC AUC para multiclase (one-vs-rest)
        try:
            roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        except:
            roc_auc = None
            print("No se pudo calcular ROC AUC para el conjunto de datos multiclase")
        
        # Retornar resultados completos
        return {
            'cv_results': random_search.cv_results_,
            'best_params': best_params,
            'best_score': random_search.best_score_,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'test_data': (X_test_scaled, y_test, y_pred, y_prob),
            'scaler': scaler
        }
  
    def save_model(self, filepath):
        """
        Guarda el modelo SVM entrenado en un archivo.
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        import joblib
        
        # Verificar que el modelo esté entrenado
        if not hasattr(self, 'svm_classifier') or self.svm_classifier is None:
            raise ValueError("El modelo SVM no ha sido entrenado. Llame a train_svm() primero.")
            
        # Guardar modelo
        joblib.dump(self.svm_classifier, filepath)
        print(f"Modelo guardado en {filepath}")
        
def predict(self, X_new, threshold=0.5):
        """
        Realiza predicciones con el modelo SVM entrenado.
        
        Args:
            X_new: DataFrame con nuevos datos
            threshold: Umbral de probabilidad para clasificación binaria
            
        Returns:
            Predicciones de clase y probabilidades
        """
        # Verificar que el modelo esté entrenado
        if not hasattr(self, 'svm_classifier') or self.svm_classifier is None:
            raise ValueError("El modelo SVM no ha sido entrenado. Llame a train_svm() primero.")
            
        # Obtener probabilidades
        probas = self.svm_classifier.predict_proba(X_new)[:, 1]
        
        # Clasificar según umbral
        predictions = (probas >= threshold).astype(int)
        
        return predictions, probas
        