"""
Módulo para la construcción y entrenamiento de modelos
"""
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
from scipy.stats import uniform, randint
from sklearn.preprocessing import StandardScaler
from config import SVM_MODELS_DIR, SVM_MODEL_FILENAME
from utils import save_model
import sys

        
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
        mode = 'binario'
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
        
        
        model = {
            'features': X.columns.tolist(),
            'trained_model': self.svm_classifier,
            'model_name': 'SVM',
            'classification_type': mode
        }
        save_model(model, mode, SVM_MODELS_DIR, SVM_MODEL_FILENAME,
                   precision_score(y_test, y_pred),recall_score(y_test, y_pred),
                   f1_score(y_test, y_pred),roc_auc)
        
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
            'test_data': ( y_test, y_prob),
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
    
        