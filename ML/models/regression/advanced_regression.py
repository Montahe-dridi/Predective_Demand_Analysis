# ML/models/regression/advanced_regression.py

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
from typing import Any, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from ML.models.base_model import BaseModel
from ML.config.model_config import ModelConfig

# Try to import XGBoost, use fallback if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available, using alternatives")

class AdvancedRegressionModel(BaseModel):
    """Advanced regression with multiple algorithms and hyperparameter tuning"""
    
    def __init__(self, algorithm: str = 'random_forest', config: ModelConfig = None):
        super().__init__(f"{algorithm}_regression", config)
        self.algorithm = algorithm
        self.best_params = None
    
    def create_model(self) -> Any:
        """Create model based on algorithm choice"""
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100, 
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=100,
                random_state=self.config.RANDOM_STATE
            ),
            'ridge': Pipeline([
                ('scaler', RobustScaler()),
                ('regressor', Ridge(random_state=self.config.RANDOM_STATE))
            ]),
            'lasso': Pipeline([
                ('scaler', RobustScaler()),
                ('regressor', Lasso(random_state=self.config.RANDOM_STATE))
            ]),
            'elastic_net': Pipeline([
                ('scaler', RobustScaler()),
                ('regressor', ElasticNet(random_state=self.config.RANDOM_STATE))
            ])
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBRegressor(
                n_estimators=100,
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1
            )
        
        return models.get(self.algorithm, models['random_forest'])
    
    def get_hyperparameter_grid(self) -> Dict:
        """Return hyperparameter grid for tuning"""
        grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            },
            'ridge': {
                'regressor__alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'lasso': {
                'regressor__alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'elastic_net': {
                'regressor__alpha': [0.1, 1.0, 10.0],
                'regressor__l1_ratio': [0.1, 0.5, 0.9]
            }
        }
        
        # Add XGBoost grid if available
        if XGBOOST_AVAILABLE:
            grids['xgboost'] = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'subsample': [0.8, 0.9, 1.0]
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
                scoring='neg_mean_squared_error',
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1
            )
        else:
            search = GridSearchCV(
                base_model, param_grid, 
                cv=5, scoring='neg_mean_squared_error',
                n_jobs=-1
            )
        
        search.fit(X, y)
        self.best_params = search.best_params_
        self.model = search.best_estimator_
        self.is_fitted = True
        
        return {
            'best_params': search.best_params_,
            'best_score': -search.best_score_,
            'cv_results': search.cv_results_
        }
    
    def get_scoring_metric(self) -> str:
        return 'neg_mean_squared_error'
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance if available"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Handle pipeline models
        if hasattr(self.model, 'named_steps') and 'regressor' in self.model.named_steps:
            model_obj = self.model.named_steps['regressor']
        else:
            model_obj = self.model
        
        # Get importance based on model type
        if hasattr(model_obj, 'feature_importances_'):
            importances = model_obj.feature_importances_
        elif hasattr(model_obj, 'coef_'):
            importances = np.abs(model_obj.coef_)
        else:
            return pd.DataFrame()
        
        feature_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_df