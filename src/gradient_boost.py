from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
from scipy.stats import uniform, randint
from config import GB_MODEL_FILENAME, GB_MODELS_DIR
from utils import save_model

class GradientBoosting:
    """
    Manages GradientBoostingClassifier configuration, training, and optimization.
    
    Handles default hyperparameter setup and cross-validation strategies.
    """
    def __init__(self):
        # Set default hyperparameters
        self.gb_n_estimators = 100
        self.gb_learning_rate = 0.1
        self.gb_max_depth = 3
        self.gb_min_samples_split = 2
        self.gb_min_samples_leaf = 1
        self.gb_subsample = 1.0
        self.gb_max_features = None
        
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
        
        # Define search space for RandomizedSearchCV
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
        print("Entrenando modelo Gradient Boosting...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.gb_classifier.fit(X_train, y_train)
        
        # Run inference
        y_pred = self.gb_classifier.predict(X_test)
        y_prob = self.gb_classifier.predict_proba(X_test)[:, 1]
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.gb_classifier, X, y, cv=cv, scoring='roc_auc'
        )

        # Calculate threshold metrics
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        model = {
            'features': X.columns.tolist(),
            'trained_model': self.gb_classifier,
            'model_name': 'GradientBoosting',
            'classification_type': mode
        }
        
        save_model(model, mode, GB_MODELS_DIR, GB_MODEL_FILENAME,
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
            'pr_curve': (precision, recall),
            'feature_importance': self.gb_classifier.feature_importances_
        }
    
    def train_gb_multiclase(self, X, y):  
        """
        Trains a multi-class classifier.
        
        Evaluates metrics using a One-vs-Rest (OvR) strategy.
        
        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Multi-class target variable.

        Returns:
            dict: Aggregated metrics (macro-average), OvR AUC, and confusion matrix.
        """
        print("Entrenando modelo Gradient Boosting Multiclase...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.gb_classifier.fit(X_train, y_train)
        
        y_pred = self.gb_classifier.predict(X_test)
        y_prob = self.gb_classifier.predict_proba(X_test)
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.gb_classifier, X, y, cv=cv, scoring='accuracy'
        )
        
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
        Optimizes hyperparameters using RandomizedSearchCV.
        
        Updates the internal `self.gb_classifier` with the best estimator found.
        
        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target variable.
            n_iter (int): Number of search iterations. Default: 50.
            cv_folds (int): Number of cross-validation folds. Default: 5.
            
        Returns:
            dict: The best hyperparameters found.
        """
        print(f"Optimizando hiperparámetros de Gradient Boosting con {n_iter} iteraciones...")
        
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
        
        random_search.fit(X, y)
        
        self.gb_classifier = random_search.best_estimator_
        
        print(f"\nMEJORES PARÁMETROS ENCONTRADOS:")
        for param, value in random_search.best_params_.items():
            print(f"  {param}: {value}")
        
        print(f"\nMEJOR PUNTUACIÓN CV: {random_search.best_score_:.4f}")
        
        return random_search.best_params_