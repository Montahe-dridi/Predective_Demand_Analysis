# ML/models/ensemble/model_ensemble.py

from sklearn.ensemble import VotingRegressor, VotingClassifier, StackingRegressor, StackingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import sys
import os
import joblib
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from ML.config.model_config import ModelConfig

class ModelEnsemble:
    """Ensemble methods for combining multiple models"""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.ensemble_model = None
        self.base_models = {}
        self.weights = None
        self.ensemble_type = None
    
    def create_voting_ensemble(self, models: Dict[str, Any], task_type: str = 'regression') -> Any:
        """Create voting ensemble (soft voting for classification)"""
        estimators = [(name, model) for name, model in models.items()]
        
        if task_type == 'regression':
            return VotingRegressor(estimators=estimators, n_jobs=-1)
        else:
            return VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
    
    def create_stacking_ensemble(self, models: Dict[str, Any], task_type: str = 'regression') -> Any:
        """Create stacking ensemble with meta-learner"""
        estimators = [(name, model) for name, model in models.items()]
        
        if task_type == 'regression':
            meta_learner = LinearRegression()
            return StackingRegressor(
                estimators=estimators, 
                final_estimator=meta_learner,
                cv=5,
                n_jobs=-1
            )
        else:
            meta_learner = LogisticRegression(random_state=self.config.RANDOM_STATE)
            return StackingClassifier(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=5,
                n_jobs=-1
            )
    
    def train_weighted_ensemble(self, X: pd.DataFrame, y: pd.Series, 
                              models: Dict[str, Any], 
                              task_type: str = 'regression',
                              ensemble_type: str = 'voting') -> Dict:
        """Train ensemble with optimal weights based on CV performance"""
        
        print(f"🔄 Training {ensemble_type} ensemble with {len(models)} models...")
        
        # Get individual model CV scores
        individual_scores = {}
        for name, model in models.items():
            try:
                if task_type == 'regression':
                    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
                    individual_scores[name] = -scores.mean()  # Convert to positive RMSE
                else:
                    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
                    individual_scores[name] = scores.mean()
                    
                print(f"  ✅ {name}: {individual_scores[name]:.4f}")
                
            except Exception as e:
                print(f"  ❌ {name} failed: {e}")
                continue
        
        if not individual_scores:
            raise ValueError("No models successfully trained")
        
        # Calculate weights (inverse for regression, direct for classification)
        if task_type == 'regression':
            weights = {name: 1/score for name, score in individual_scores.items()}
        else:
            weights = individual_scores.copy()
        
        # Normalize weights
        weight_sum = sum(weights.values())
        self.weights = {name: w/weight_sum for name, w in weights.items()}
        
        # Create and train ensemble
        if ensemble_type == 'voting':
            self.ensemble_model = self.create_voting_ensemble(models, task_type)
        elif ensemble_type == 'stacking':
            self.ensemble_model = self.create_stacking_ensemble(models, task_type)
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")
        
        self.ensemble_model.fit(X, y)
        self.ensemble_type = ensemble_type
        self.base_models = models
        
        # Evaluate ensemble performance
        if task_type == 'regression':
            ensemble_scores = cross_val_score(self.ensemble_model, X, y, cv=5, scoring='neg_mean_squared_error')
            ensemble_performance = -ensemble_scores.mean()
        else:
            ensemble_scores = cross_val_score(self.ensemble_model, X, y, cv=5, scoring='accuracy')
            ensemble_performance = ensemble_scores.mean()
        
        return {
            'individual_scores': individual_scores,
            'weights': self.weights,
            'ensemble_performance': ensemble_performance,
            'ensemble_cv_scores': ensemble_scores.tolist(),
            'ensemble_type': ensemble_type,
            'best_individual': max(individual_scores.items(), key=lambda x: x[1] if task_type == 'classification' else 1/x[1])
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions"""
        if self.ensemble_model is None:
            raise ValueError("Ensemble must be trained first")
        return self.ensemble_model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get ensemble prediction probabilities (classification only)"""
        if self.ensemble_model is None:
            raise ValueError("Ensemble must be trained first")
        
        if hasattr(self.ensemble_model, 'predict_proba'):
            return self.ensemble_model.predict_proba(X)
        else:
            raise ValueError("Ensemble doesn't support probability predictions")
    
    def get_ensemble_weights(self) -> Dict[str, float]:
        """Get the weights used in the ensemble"""
        return self.weights or {}
    
    def save_ensemble(self, filepath: str = None) -> str:
        """Save ensemble model"""
        if filepath is None:
            filepath = os.path.join(self.config.MODEL_DIR, f"ensemble_{self.ensemble_type}.joblib")
        
        ensemble_data = {
            'ensemble_model': self.ensemble_model,
            'base_models': self.base_models,
            'weights': self.weights,
            'ensemble_type': self.ensemble_type,
            'config': self.config
        }
        
        joblib.dump(ensemble_data, filepath)
        return filepath