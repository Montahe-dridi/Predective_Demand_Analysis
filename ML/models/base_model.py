# ML/models/base_model.py

from abc import ABC, abstractmethod
import joblib
import pandas as pd
from typing import Any, Dict, Tuple
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ML.config.model_config import ModelConfig

class BaseModel(ABC):
    """Abstract base class for all ML models"""
    
    def __init__(self, model_name: str, config: ModelConfig = None):
        self.model_name = model_name
        self.config = config or ModelConfig()
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.performance_history = []
    
    @abstractmethod
    def create_model(self) -> Any:
        """Create the underlying ML model"""
        pass
    
    @abstractmethod
    def get_hyperparameter_grid(self) -> Dict:
        """Return hyperparameter grid for tuning"""
        pass
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Fit model with cross-validation"""
        self.model = self.create_model()
        if hasattr(self.model, 'max_depth'):
           self.model.max_depth = min(self.model.max_depth or 20, 15)
        if hasattr(self.model, 'min_samples_split'):
           self.model.min_samples_split = max(self.model.min_samples_split, 10)
        self.feature_names = X.columns.tolist()
        
        # Cross-validation for time series data
        if 'Date' in str(X.index) or any('date' in col.lower() for col in X.columns):
            cv = TimeSeriesSplit(n_splits=self.config.CV_FOLDS)
        else:
            cv = self.config.CV_FOLDS
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring=self.get_scoring_metric())
        
        # Final fit on all data
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Store performance
        performance = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
        self.performance_history.append(performance)
        
        return performance
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def save_model(self) -> str:
        """Save model to disk"""
        path = os.path.join(self.config.MODEL_DIR, f"{self.model_name}.joblib")
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'performance_history': self.performance_history,
            'config': self.config
        }
        joblib.dump(model_data, path)
        return path
    
    def load_model(self, path: str = None) -> bool:
        """Load model from disk"""
        if path is None:
            path = os.path.join(self.config.MODEL_DIR, f"{self.model_name}.joblib")
        
        try:
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.performance_history = model_data['performance_history']
            self.is_fitted = True
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    @abstractmethod
    def get_scoring_metric(self) -> str:
        """Return appropriate scoring metric"""
        pass