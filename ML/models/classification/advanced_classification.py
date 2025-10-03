# ML/models/classification/advanced_classification.py

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
import numpy as np
from typing import Any, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from ML.models.base_model import BaseModel
from ML.config.model_config import ModelConfig

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class AdvancedClassificationModel(BaseModel):
    """Advanced classification with multiple algorithms"""
    
    def __init__(self, algorithm: str = 'random_forest', config: ModelConfig = None):
        super().__init__(f"{algorithm}_classification", config)
        self.algorithm = algorithm
        self.class_labels = None
        self.best_params = None
    
    def create_model(self) -> Any:
        """Create classification model"""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.config.RANDOM_STATE
            ),
            'logistic': Pipeline([
                ('scaler', RobustScaler()),
                ('classifier', LogisticRegression(
                    random_state=self.config.RANDOM_STATE,
                    class_weight='balanced',
                    max_iter=1000
                ))
            ]),
            'svm': Pipeline([
                ('scaler', RobustScaler()),
                ('classifier', SVC(
                    random_state=self.config.RANDOM_STATE,
                    probability=True,
                    class_weight='balanced'
                ))
            ])
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1
            )
        
        return models.get(self.algorithm, models['random_forest'])
    
    def get_hyperparameter_grid(self) -> Dict:
        """Return hyperparameter grid for classification"""
        grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'class_weight': ['balanced', None]
            },
            'gradient_boost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            },
            'logistic': {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__penalty': ['l2', 'l1'],
                'classifier__solver': ['liblinear', 'saga']
            },
            'svm': {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__kernel': ['rbf', 'poly', 'linear']
            }
        }
        
        # Add XGBoost grid if available
        if XGBOOST_AVAILABLE:
            grids['xgboost'] = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            }
        
        return grids.get(self.algorithm, {})
    
    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series,
                           search_type: str = 'random') -> Dict:
        """Perform hyperparameter tuning"""
        param_grid = self.get_hyperparameter_grid()
        
        if not param_grid:
            print(f"No hyperparameter grid for {self.algorithm}")
            return {}
        
        base_model = self.create_model()
        
        if search_type == 'random':
            search = RandomizedSearchCV(
                base_model, param_grid,
                n_iter=20, cv=5,
                scoring='accuracy',
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1
            )
        else:
            search = GridSearchCV(
                base_model, param_grid,
                cv=5, scoring='accuracy',
                n_jobs=-1
            )
        
        search.fit(X, y)
        self.best_params = search.best_params_
        self.model = search.best_estimator_
        self.is_fitted = True
        
        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_
        }
    
    def get_scoring_metric(self) -> str:
        return 'accuracy'
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError("Model doesn't support probability predictions")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance if available"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Handle pipeline models
        if hasattr(self.model, 'named_steps') and 'classifier' in self.model.named_steps:
            model_obj = self.model.named_steps['classifier']
        else:
            model_obj = self.model
        
        # Get importance based on model type
        if hasattr(model_obj, 'feature_importances_'):
            importances = model_obj.feature_importances_
        elif hasattr(model_obj, 'coef_'):
            # For multi-class, take mean of absolute coefficients
            coef = model_obj.coef_
            if coef.ndim > 1:
                importances = np.mean(np.abs(coef), axis=0)
            else:
                importances = np.abs(coef)
        else:
            return pd.DataFrame()
        
        feature_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_df