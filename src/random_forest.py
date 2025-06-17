"""
Módulo para la construcción y entrenamiento de modelos Random Forest
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
from scipy.stats import randint
from config import RF_MODEL_FILENAME, RF_MODELS_DIR
from utils import save_model
        
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
        mode = 'binario'
        
        
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
        
        
        model = {
            'features': X.columns.tolist(),
            'trained_model': self.rf_classifier,
            'model_name': 'RandomForest',
            'classification_type': mode
        }
        save_model(model, mode, RF_MODELS_DIR, RF_MODEL_FILENAME,
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
            'test_data': (y_test, y_prob),
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

