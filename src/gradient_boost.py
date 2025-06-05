"""
Módulo para la construcción y entrenamiento de modelos Gradient Boosting
"""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score

from scipy.stats import uniform, randint
import pandas as pd
from config import *

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import ConfusionMatrixDisplay

class GradientBoost:
    def __init__(self):
        # Parámetros optimizados para Gradient Boosting
        """
        Configuración optimizada para Gradient Boosting basada en mejores prácticas
        """
        
        self.gb_n_estimators = 100
        self.gb_learning_rate = 0.1
        self.gb_max_depth = 3
        self.gb_min_samples_split = 2
        self.gb_min_samples_leaf = 1
        self.gb_subsample = 1.0
        self.gb_max_features = None
        
        # Inicializar clasificador con parámetros optimizados
        self.gb_classifier = GradientBoostingClassifier(
            n_estimators=self.gb_n_estimators,
            learning_rate=self.gb_learning_rate,
            max_depth=self.gb_max_depth,
            min_samples_split=self.gb_min_samples_split,
            min_samples_leaf=self.gb_min_samples_leaf,
            subsample=self.gb_subsample,
            max_features=self.gb_max_features,
            random_state=42,
            verbose=0
        )
        
        # Distribuciones de parámetros para búsqueda aleatoria
        self.param_distributions = {
            'n_estimators': randint(50, 300),
            'learning_rate': uniform(0.01, 0.3),
            'max_depth': randint(3, 10),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'subsample': uniform(0.6, 0.4),
            'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7]
        }
                      
    def train_gb_binary(self, X, y):  
        """
        Entrena el modelo de clasificación Gradient Boosting con opciones para manejar desbalance de clases.
        
        Args:
            X: Features
            y: Variable objetivo
            
        Returns:
            Diccionario con resultados del entrenamiento
        """
        print("Entrenando modelo Gradient Boosting...")
        
        # División del dataset con estratificación
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Entrenamiento del modelo
        self.gb_classifier.fit(X_train, y_train)
        
        # Evaluación
        y_pred = self.gb_classifier.predict(X_test)
        y_prob = self.gb_classifier.predict_proba(X_test)[:, 1]
        
        # Validación cruzada
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.gb_classifier, X, y, cv=cv, scoring='roc_auc'
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
            'feature_importance': self.gb_classifier.feature_importances_
        }
    
    def train_gb_multiclase(self, X, y):  
        """
        Entrena el modelo de clasificación Gradient Boosting multiclase con opciones para manejar desbalance de clases.
        
        Args:
            X: Features
            y: Variable objetivo

            
        Returns:
            Diccionario con resultados del entrenamiento
        """
        print("Entrenando modelo Gradient Boosting Multiclase...")
        
        # División del dataset con estratificación
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        
        # Entrenamiento del modelo
        self.gb_classifier.fit(X_train, y_train)
        
        # Evaluación
        y_pred = self.gb_classifier.predict(X_test)
        y_prob = self.gb_classifier.predict_proba(X_test)
        
        # Validación cruzada con métrica adecuada para multiclase
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.gb_classifier, X, y, cv=cv, scoring='accuracy'
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
            'feature_importance': self.gb_classifier.feature_importances_
        }
       
    def mostrar_resultados_gb_binario(self, resultados):
        """Muestra los resultados del entrenamiento del Gradient Boosting con más detalle."""
        print("\n========== RESULTADOS DEL ENTRENAMIENTO GRADIENT BOOSTING ==========")
        
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
        
        # Importancia de características (específico de Gradient Boosting)
        if 'feature_importance' in resultados:
            print("\n🚀 IMPORTANCIA DE CARACTERÍSTICAS (Top 10):")
            importances = resultados['feature_importance']
            top_indices = importances.argsort()[-10:][::-1]
            for i, idx in enumerate(top_indices):
                print(f"  {i+1}. Característica {idx}: {importances[idx]:.4f}")
    
    def mostrar_resultados_gb_multiclase(self, resultados):
        """Muestra los resultados del entrenamiento del Gradient Boosting multiclase."""
        print("\n========== RESULTADOS DEL ENTRENAMIENTO GRADIENT BOOSTING MULTICLASE ==========")
            
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
        
        # Importancia de características (específico de Gradient Boosting)
        if 'feature_importance' in resultados:
            print("\n🚀 IMPORTANCIA DE CARACTERÍSTICAS (Top 10):")
            importances = resultados['feature_importance']
            top_indices = importances.argsort()[-10:][::-1]
            for i, idx in enumerate(top_indices):
                print(f"  {i+1}. Característica {idx}: {importances[idx]:.4f}")
    
    def optimize_gb_hyperparameters(self, X, y, n_iter=50, cv_folds=5):
        """
        Optimiza los hiperparámetros del modelo Gradient Boosting usando RandomizedSearchCV.
        
        Args:
            X: Features
            y: Variable objetivo
            n_iter: Número de iteraciones para la búsqueda aleatoria
            cv_folds: Número de folds para validación cruzada
            
        Returns:
            Mejores parámetros encontrados
        """
        print(f"Optimizando hiperparámetros de Gradient Boosting con {n_iter} iteraciones...")
        
        # Configurar búsqueda aleatoria
        random_search = RandomizedSearchCV(
            estimator=self.gb_classifier,
            param_distributions=self.param_distributions,
            n_iter=n_iter,
            cv=cv_folds,
            scoring='roc_auc',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        # Realizar búsqueda
        random_search.fit(X, y)
        
        # Actualizar clasificador con mejores parámetros
        self.gb_classifier = random_search.best_estimator_
        
        print(f"\n🎯 MEJORES PARÁMETROS ENCONTRADOS:")
        for param, value in random_search.best_params_.items():
            print(f"  {param}: {value}")
        
        print(f"\n📊 MEJOR PUNTUACIÓN CV: {random_search.best_score_:.4f}")
        
        return random_search.best_params_
    
    def get_feature_importance_df(self, feature_names):
        """
        Obtiene la importancia de características como DataFrame ordenado.
        
        Args:
            feature_names: Lista de nombres de las características
            
        Returns:
            DataFrame con importancia de características ordenado
        """
        if hasattr(self.gb_classifier, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.gb_classifier.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            print("El modelo no ha sido entrenado aún.")
            return None

    def visualizar_resultados_gb_binario(self, resultados, nombre_modelo='GradientBoost_Binary'):
        """
        Visualiza los resultados del modelo Gradient Boosting binario.
        
        Args:
            resultados: Diccionario con los resultados del entrenamiento
            nombre_modelo: Nombre del modelo para usar en los archivos guardados
        """
        # Crear directorio para visualizaciones si no existe
        if not os.path.exists('visual_model'):
            os.makedirs('visual_model')
        
        # Obtener datos de prueba
        X_test_scaled, y_test, y_pred, y_prob = resultados['test_data']
        
        # 1. Métricas del modelo en gráfico de barras
        plt.figure(figsize=(12, 6))
        metricas = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        valores = [resultados[metrica] for metrica in metricas]
        
        sns.barplot(x=metricas, y=valores)
        plt.title(f'Métricas del modelo {nombre_modelo}')
        plt.ylim(0, 1)
        for i, v in enumerate(valores):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'visual_model/metricas_{nombre_modelo}.png')
        plt.close()
        
        # 2. Matriz de confusión
        plt.figure(figsize=(8, 6))
        cm = resultados['confusion_matrix']
        clases = np.unique(y_test)
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)
        disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
        plt.title(f'Matriz de Confusión - {nombre_modelo}')
        plt.savefig(f'visual_model/matriz_confusion_{nombre_modelo}.png')
        plt.close()
        
        # 3. Curva ROC
        plt.figure(figsize=(8, 6))
        fpr, tpr = resultados['roc_curve']
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {resultados["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title(f'Curva ROC - {nombre_modelo}')
        plt.legend(loc="lower right")
        plt.savefig(f'visual_model/curva_roc_{nombre_modelo}.png')
        plt.close()
        
        # 4. Curva Precision-Recall
        plt.figure(figsize=(8, 6))
        precision, recall = resultados['pr_curve']
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AUC = {resultados["pr_auc"]:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Curva Precision-Recall - {nombre_modelo}')
        plt.legend()
        plt.savefig(f'visual_model/curva_pr_{nombre_modelo}.png')
        plt.close()
        
        # 5. Distribución de probabilidades
        plt.figure(figsize=(10, 6))
        for clase in np.unique(y_test):
            mask = y_test == clase
            if np.any(mask):
                sns.kdeplot(y_prob[mask], label=f'Clase {clase}', alpha=0.7)
        
        plt.xlabel('Probabilidad predicha')
        plt.ylabel('Densidad')
        plt.title(f'Distribución de probabilidades por clase - {nombre_modelo}')
        plt.legend()
        plt.savefig(f'visual_model/distribucion_probabilidades_{nombre_modelo}.png')
        plt.close()
        
        # 6. Validación cruzada
        plt.figure(figsize=(10, 6))
        cv_scores = resultados['cv_scores']
        plt.boxplot(cv_scores)
        plt.title(f'Distribución de puntuaciones en validación cruzada - {nombre_modelo}')
        plt.ylabel('ROC AUC')
        plt.axhline(y=np.mean(cv_scores), color='r', linestyle='--', 
                    label=f'Media: {np.mean(cv_scores):.3f}')
        plt.text(1.1, np.mean(cv_scores), f'{np.mean(cv_scores):.3f}', color='r')
        plt.savefig(f'visual_model/cv_scores_{nombre_modelo}.png')
        plt.close()
        
        # 7. Importancia de características
        if 'feature_importance' in resultados:
            plt.figure(figsize=(12, 8))
            importances = resultados['feature_importance']
            top_indices = importances.argsort()[-15:][::-1]
            
            plt.barh(range(len(top_indices)), importances[top_indices])
            plt.yticks(range(len(top_indices)), [f'Feature {i}' for i in top_indices])
            plt.xlabel('Importancia')
            plt.title(f'Top 15 Características más importantes - {nombre_modelo}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f'visual_model/importancia_caracteristicas_{nombre_modelo}.png')
            plt.close()
        
        # 8. Resumen del clasificador
        with open(f'reporte_{nombre_modelo}.txt', 'w') as f:
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
        
        print(f"Visualizaciones guardadas en el directorio 'visual_model/'")
        return nombre_modelo

    def visualizar_resultados_gb_multiclase(self, resultados, nombre_modelo='GradientBoost_Multiclass'):
        """
        Visualiza los resultados del modelo Gradient Boosting multiclase.
        
        Args:
            resultados: Diccionario con los resultados del entrenamiento
            nombre_modelo: Nombre del modelo para usar en los archivos guardados
        """
        # Crear directorio para visualizaciones si no existe
        if not os.path.exists('visual_model'):
            os.makedirs('visual_model')
        
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
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'visual_model/metricas_{nombre_modelo}.png')
        plt.close()
        
        # 2. Matriz de confusión
        plt.figure(figsize=(10, 8))
        cm = resultados['confusion_matrix']
        clases = np.unique(y_test)
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)
        disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
        plt.title(f'Matriz de Confusión - {nombre_modelo}')
        plt.savefig(f'visual_model/matriz_confusion_{nombre_modelo}.png')
        plt.close()
        
        # 3. Distribución de probabilidades por clase
        plt.figure(figsize=(15, 10))
        n_clases = y_prob.shape[1]
        
        for i in range(n_clases):
            plt.subplot(2, (n_clases + 1) // 2, i+1)
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
        plt.savefig(f'visual_model/distribucion_probabilidades_{nombre_modelo}.png')
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
        plt.savefig(f'visual_model/cv_scores_{nombre_modelo}.png')
        plt.close()
        
        # 5. Importancia de características
        if 'feature_importance' in resultados:
            plt.figure(figsize=(12, 8))
            importances = resultados['feature_importance']
            top_indices = importances.argsort()[-15:][::-1]
            
            plt.barh(range(len(top_indices)), importances[top_indices])
            plt.yticks(range(len(top_indices)), [f'Feature {i}' for i in top_indices])
            plt.xlabel('Importancia')
            plt.title(f'Top 15 Características más importantes - {nombre_modelo}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f'visual_model/importancia_caracteristicas_{nombre_modelo}.png')
            plt.close()
        
        # 6. Visualización de clasificación por pares de características importantes
        if 'feature_importance' in resultados:
            from itertools import combinations
            
            # Obtener las características más importantes
            feature_importance = resultados['feature_importance']
            top_features_idx = np.argsort(feature_importance)[-5:] if len(feature_importance) > 5 else np.argsort(feature_importance)
            
            # Crear gráficos de dispersión para pares de características importantes
            pairs = list(combinations(top_features_idx, 2))
            
            if pairs:
                for i, (idx1, idx2) in enumerate(pairs[:3]):  # Limitar a 3 gráficos
                    plt.figure(figsize=(10, 8))
                    
                    scatter = plt.scatter(X_test_scaled[:, idx1], X_test_scaled[:, idx2], 
                                        c=y_test, cmap='viridis', alpha=0.7, 
                                        edgecolors='w', s=100)
                    
                    # Añadir predicciones incorrectas marcadas con X
                    mask_incorrect = y_test != y_pred
                    if np.any(mask_incorrect):
                        plt.scatter(X_test_scaled[mask_incorrect, idx1], 
                                X_test_scaled[mask_incorrect, idx2],
                                marker='x', c='red', s=150, linewidths=2, 
                                label='Predicciones incorrectas')
                    
                    plt.colorbar(scatter, label='Clase')
                    plt.title(f'Clasificación en características importantes ({idx1} vs {idx2})')
                    plt.xlabel(f'Característica {idx1}')
                    plt.ylabel(f'Característica {idx2}')
                    plt.legend()
                    plt.savefig(f'visual_model/dispersion_caracteristicas_{idx1}_{idx2}_{nombre_modelo}.png')
                    plt.close()
        
        # 7. Resumen del clasificador
        with open(f'visual_model/reporte_{nombre_modelo}.txt', 'w') as f:
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
        
        print(f"Visualizaciones guardadas en el directorio 'visual_model/'")
        return nombre_modelo