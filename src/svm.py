import sys
from scipy.stats import uniform, randint
from sklearn.svm import SVC  
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, StratifiedKFold
from config import SVM_MODELS_DIR, SVM_MODEL_FILENAME
from utils import save_model

class SVM:
    """
    Manages Support Vector Machine (SVM) configuration, training, and optimization.
    
    Handles binary and multi-class classification with built-in hyperparameter 
    tuning and metric reporting.
    """
    def __init__(self):
        # Set default hyperparameters
        self.svm_c_parameter = 10.496128632644757  
        self.svm_kernel_parameter = 'poly'  
        self.svm_gamma_parameter = 2.9123914019804196 
        self.svm_tolerance_parameter = 0.01
        self.svm_class_weight_parameter = None 
        
        # Initialize estimator
        self.svm_classifier = SVC(
            kernel=self.svm_kernel_parameter, 
            probability=True, 
            C=self.svm_c_parameter, 
            tol=self.svm_tolerance_parameter, 
            class_weight=self.svm_class_weight_parameter, 
            gamma=self.svm_gamma_parameter, 
            degree=2,
            cache_size=1000
        )
        self.svm_precision_result = None
        
        # Define search space for RandomizedSearchCV
        self.param_distributions = {
            'C': uniform(0.001, 30.0),
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': uniform(0.0001, 20.0),
            'class_weight': ['balanced', None],
            'degree': randint(2, 6),
            'tol': [0.0001, 0.001, 0.01]
        }                      
                        
    def train_svm_binary(self, X, y):  
        """
        Trains a binary classifier and saves the result.
        
        Performs stratified splitting, cross-validation, and ROC/PR calculation.
        Saves the model artifact using `utils.save_model`.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Binary target variable.
            
        Returns:
            dict: Performance metrics (AUC, F1, Accuracy), reports, and test data.
        """
        mode = 'binario'
        print("Entrenando modelo SVM...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.svm_classifier.fit(X_train, y_train)
        
        # Run inference
        y_pred = self.svm_classifier.predict(X_test)
        y_prob = self.svm_classifier.predict_proba(X_test)[:, 1] 
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.svm_classifier, X, y, cv=cv, scoring='roc_auc'
        )

        # Calculate threshold metrics
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        model = {
            'features': X.columns.tolist(),
            'trained_model': self.svm_classifier,
            'model_name': 'SVM',
            'classification_type': mode
        }
        
        save_model(model, mode, SVM_MODELS_DIR, SVM_MODEL_FILENAME,
                   precision_score(y_test, y_pred), recall_score(y_test, y_pred),
                   f1_score(y_test, y_pred), roc_auc)
        
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
            'test_data': (y_test, y_prob),
            'roc_curve': (fpr, tpr),
            'pr_curve': (precision, recall)
        }
    
    def train_svm_multiclase(self, X, y):  
        """
        Trains a multi-class classifier using One-vs-Rest (OvR).
        
        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Multi-class target variable.

        Returns:
            dict: Aggregated metrics (macro-average), OvR AUC, and confusion matrix.
        """
        print("Entrenando modelo SVM Multiclase...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.svm_classifier.fit(X_train, y_train)
        
        y_pred = self.svm_classifier.predict(X_test)
        y_prob = self.svm_classifier.predict_proba(X_test)
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.svm_classifier, X, y, cv=cv, scoring='accuracy' 
        )

        # Calculate Multiclass AUC
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
            'test_data': (X_test, y_test, y_pred, y_prob)
        } 
    
    def find_best_parameters_binary(self, X, y, n_iter=50, cv_folds=5, scoring='recall'):
        """
        Optimizes hyperparameters using RandomizedSearchCV.
        
        Updates the internal `self.svm_classifier` with the best estimator found.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target variable.
            n_iter (int): Number of search iterations.
            cv_folds (int): Number of cross-validation folds.
            scoring (str): Target metric (e.g., 'roc_auc', 'f1').
            
        Returns:
            dict: Detailed search results and best model metrics.
        """
        print(f"Optimizando hiperparámetros con {n_iter} iteraciones...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
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
        
        random_search.fit(X_train, y_train)
        
        self.svm_classifier = random_search.best_estimator_
        
        # Update class attributes
        best_params = random_search.best_params_
        if 'C' in best_params: self.svm_c_parameter = best_params['C']
        if 'kernel' in best_params: self.svm_kernel_parameter = best_params['kernel']
        if 'gamma' in best_params: self.svm_gamma_parameter = best_params['gamma']
        if 'class_weight' in best_params: self.svm_class_weight_parameter = best_params['class_weight']
        
        # Evaluate
        y_pred = self.svm_classifier.predict(X_test)
        y_prob = self.svm_classifier.predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        return {
            'cv_results': random_search.cv_results_,
            'best_params': best_params,
            'best_score': random_search.best_score_,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'test_data': (X_test, y_test, y_pred, y_prob),
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'roc_curve': (fpr, tpr),
            'pr_curve': (precision, recall)
        }

    def find_best_parameters_multiclass(self, X, y, n_iter=40, cv_folds=5, scoring='recall_macro'):
        """
        Optimizes hyperparameters for multi-class targets.
        
        Includes feature scaling (StandardScaler) within the optimization pipeline.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target variable.
            scoring (str): Multi-class metric (e.g., 'f1_macro').
            
        Returns:
            dict: Search results and detailed metrics for the best estimator.
        """
        print(f"Optimizando hiperparámetros para clasificación multiclase con {n_iter} iteraciones...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
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
        
        random_search.fit(X_train_scaled, y_train)
        
        self.svm_classifier = random_search.best_estimator_
        
        best_params = random_search.best_params_
        if 'C' in best_params: self.svm_c_parameter = best_params['C']
        if 'kernel' in best_params: self.svm_kernel_parameter = best_params['kernel']
        if 'gamma' in best_params: self.svm_gamma_parameter = best_params['gamma']
        if 'class_weight' in best_params: self.svm_class_weight_parameter = best_params['class_weight']
        
        y_pred = self.svm_classifier.predict(X_test_scaled)
        y_prob = self.svm_classifier.predict_proba(X_test_scaled)
        
        # Multi-class metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        
        try:
            roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        except:
            roc_auc = None
            print("No se pudo calcular ROC AUC para el conjunto de datos multiclase")
        
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
    
    def mostrar_resultados_randomized_search_binary(self, resultados, nombre_archivo="svm_randomizer_search_results.txt"):
        """
        Exports optimization results to a text file.
        
        Temporarily redirects stdout to capture the detailed report.

        Args:
            resultados (dict): Results from `find_best_parameters_binary`.
            nombre_archivo (str): Output filename.
        """
        
        with open(nombre_archivo, 'w', encoding='utf-8') as f:
            sys.stdout = f

            print("Muestra los resultados mejorados de RandomizedSearchCV.")
            print("\n========== RESULTADOS DE RANDOMIZED SEARCH CV ==========")
    
            print("\nMEJORES PARÁMETROS:")
            for param, value in resultados['best_params'].items():
                print(f"  {param}: {value}")
            
            print(f"\nMEJOR PUNTUACIÓN: {resultados['best_score']:.4f}")
            
            print("\nTOP 5 COMBINACIONES DE PARÁMETROS:")
            cv_results = resultados['cv_results']
            sorted_indices = cv_results['mean_test_score'].argsort()[::-1][:5]
            
            for i, idx in enumerate(sorted_indices):
                mean = cv_results['mean_test_score'][idx]
                std = cv_results['std_test_score'][idx]
                params = cv_results['params'][idx]
                print(f"  {i+1}. Score: {mean:.4f} (±{std:.4f})")
                for param, value in params.items():
                    print(f"     {param}: {value}")
                print("")
            
            if 'roc_auc' in resultados:
                print(f"\nROC AUC en test: {resultados['roc_auc']:.4f}")
            if 'pr_auc' in resultados:
                print(f"PR AUC en test: {resultados['pr_auc']:.4f}")
            
            print("\nINFORME DE CLASIFICACIÓN:")
            print(resultados['classification_report'])
            
            print("\nMATRIZ DE CONFUSIÓN:")
            cm = resultados['confusion_matrix']
            print(f"  [ {cm[0][0]}\t{cm[0][1]} ]")
            print(f"  [ {cm[1][0]}\t{cm[1][1]} ]")

        sys.stdout = sys.__stdout__
        print(f"Resultados guardados exitosamente en '{nombre_archivo}'")
        
    def mostrar_resultados_randomized_search_multiclase(self, resultados):
        """
        Prints detailed multi-class performance metrics to the console.

        Args:
            resultados (dict): Results from `find_best_parameters_multiclass`.
        """
        print("\n========== RESULTADOS DE RANDOMIZED SEARCH CV MULTICLASE ==========")
        
        print("\nMEJORES PARÁMETROS:")
        for param, value in resultados['best_params'].items():
            print(f"  {param}: {value}")
        
        print(f"\nMEJOR PUNTUACIÓN: {resultados['best_score']:.4f}")
        
        print("\nTOP 5 COMBINACIONES DE PARÁMETROS:")
        cv_results = resultados['cv_results']
        sorted_indices = cv_results['mean_test_score'].argsort()[::-1][:5]
        
        for i, idx in enumerate(sorted_indices):
            mean = cv_results['mean_test_score'][idx]
            std = cv_results['std_test_score'][idx]
            params = cv_results['params'][idx]
            print(f"  {i+1}. Score: {mean:.4f} (±{std:.4f})")
            for param, value in params.items():
                print(f"     {param}: {value}")
            print("")
        
        print("\nMÉTRICAS DE EVALUACIÓN EN TEST:")
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
        
        print("\nINFORME DE CLASIFICACIÓN:")
        print(resultados['classification_report'])
        
        print("\nMATRIZ DE CONFUSIÓN:")
        cm = resultados['confusion_matrix']
        
        n_classes = len(cm)
        
        print("  ", end="")
        for i in range(n_classes):
            print(f"[{i}]", end="\t")
        print("")
        
        for i in range(n_classes):
            print(f"  [{i}]", end=" ")
            for j in range(n_classes):
                print(f"{cm[i][j]}", end="\t")
            print("")
        
        print("\nANÁLISIS DE LA MATRIZ DE CONFUSIÓN:")
        for i in range(n_classes):
            total_true = sum(cm[i])
            if total_true > 0:
                correct = cm[i][i]
                print(f"  Clase {i}: {correct}/{total_true} correctamente clasificados ({correct/total_true*100:.1f}%)")
                
                if n_classes > 2:  
                    confusions = [(j, cm[i][j]) for j in range(n_classes) if j != i and cm[i][j] > 0]
                    if confusions:
                        confusions.sort(key=lambda x: x[1], reverse=True)
                        print(f"    Principales confusiones: ", end="")
                        for cls, count in confusions[:2]:  
                            print(f"con clase {cls} ({count} casos, {count/total_true*100:.1f}%)", end="; ")
                        print("")