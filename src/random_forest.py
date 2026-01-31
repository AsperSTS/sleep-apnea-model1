from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
from scipy.stats import randint
from config import RF_MODEL_FILENAME, RF_MODELS_DIR
from utils import save_model
        
class RandomForest:
    """
    Manages RandomForestClassifier configuration, training, and optimization.
    
    Handles default hyperparameter setup and cross-validation strategies.
    """
    
    def __init__(self):
        # Set default hyperparameters
        self.rf_n_estimators = 100
        self.rf_max_depth = None
        self.rf_min_samples_split = 2
        self.rf_min_samples_leaf = 1
        self.rf_max_features = 'sqrt'
        self.rf_bootstrap = True
        self.rf_class_weight = 'balanced'
        
        # Initialize estimator
        self.rf_classifier = RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            min_samples_split=self.rf_min_samples_split,
            min_samples_leaf=self.rf_min_samples_leaf,
            max_features=self.rf_max_features,
            bootstrap=self.rf_bootstrap,
            class_weight=self.rf_class_weight,
            random_state=42,
            n_jobs=-1 
        )
        
        # Define search space for RandomizedSearchCV
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
        print("Entrenando modelo Random Forest...")
        
        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.rf_classifier.fit(X_train, y_train)
        
        # Run inference
        y_pred = self.rf_classifier.predict(X_test)
        y_prob = self.rf_classifier.predict_proba(X_test)[:, 1]
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.rf_classifier, X, y, cv=cv, scoring='roc_auc'
        )

        # Calculate threshold metrics
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        # Save model artifact
        model = {
            'features': X.columns.tolist(),
            'trained_model': self.rf_classifier,
            'model_name': 'RandomForest',
            'classification_type': mode
        }
        save_model(model, mode, RF_MODELS_DIR, RF_MODEL_FILENAME,
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
            'feature_importance': self.rf_classifier.feature_importances_
        }
    
    def train_rf_multiclase(self, X, y):  
        """
        Trains a multi-class classifier.
        
        Evaluates metrics using a One-vs-Rest (OvR) strategy.
        
        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Multi-class target variable.
            
        Returns:
            dict: Aggregated metrics (macro-average), OvR AUC, and confusion matrix.
        """
        print("Entrenando modelo Random Forest Multiclase...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.rf_classifier.fit(X_train, y_train)
        
        y_pred = self.rf_classifier.predict(X_test)
        y_prob = self.rf_classifier.predict_proba(X_test)
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.rf_classifier, X, y, cv=cv, scoring='accuracy'
        )
        
        # Calculate Multiclass AUC (One-vs-Rest)
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