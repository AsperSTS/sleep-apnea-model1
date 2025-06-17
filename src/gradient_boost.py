"""
Módulo para la construcción y entrenamiento de modelos Gradient Boosting
"""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
from scipy.stats import uniform, randint
from config import GB_MODEL_FILENAME, GB_MODELS_DIR
from utils import save_model

class GradientBoosting:
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
        mode = 'binario'
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
        
        
        model = {
            'features': X.columns.tolist(),
            'trained_model': self.gb_classifier,
            'model_name': 'GradientBoosting',
            'classification_type': mode
        }
        save_model(model, mode, GB_MODELS_DIR, GB_MODEL_FILENAME,
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