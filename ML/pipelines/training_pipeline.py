# ML/pipelines/training_pipeline.py:
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import os
import sys

# Add the ML directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ML.config.model_config import ModelConfig
from ML.features.engineering import FeatureEngineer
from ML.models.regression.advanced_regression import AdvancedRegressionModel
from ML.models.classification.advanced_classification import AdvancedClassificationModel
from ML.evaluation.metrics import ModelEvaluator

class MLTrainingPipeline:
    """Comprehensive ML training pipeline with data type detection"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.results = {}
        self.trained_models = {}
        self.feature_engineer = FeatureEngineer()
        
    def detect_data_type(self, df: pd.DataFrame) -> str:
        """Detect whether data is shipment or invoice type"""
        
        shipment_indicators = ['ShipmentDate', 'TotalWeight', 'TotalVolume', 'ShipmentValue']
        invoice_indicators = ['InvoiceDate', 'NetAmount', 'TaxAmount', 'InvoiceID']
        
        shipment_score = sum(1 for col in shipment_indicators if col in df.columns)
        invoice_score = sum(1 for col in invoice_indicators if col in df.columns)
        
        if shipment_score > invoice_score:
            return 'shipment'
        elif invoice_score > shipment_score:
            return 'invoice'
        else:
            return 'unknown'
    
    def run_regression_pipeline(self, df: pd.DataFrame, target_col: str,
                              algorithms: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive regression pipeline with auto data type detection"""
        
        if algorithms is None:
            algorithms = ['random_forest', 'gradient_boost', 'ridge']
        
        print(f"🚀 Starting regression pipeline for {target_col}")
        print(f"🎯 Algorithms to test: {algorithms}")
        
        # Detect data type
        data_type = self.detect_data_type(df)
        print(f"🔧 Engineering features...")
        
        # Apply appropriate feature engineering
        if data_type == 'shipment':
            df_engineered = self.feature_engineer.engineer_shipment_features(df)
        elif data_type == 'invoice':
            df_engineered = self.feature_engineer.engineer_invoice_features(df)
        else:
            print("⚠️ Unknown data type, using safe preprocessing")
            df_engineered = self.feature_engineer._safe_preprocessing(df)
        
        print(f"✅ Created {len(df_engineered.columns)} features")
        
        # Handle missing target
        if target_col not in df_engineered.columns:
            df_engineered = self._create_target_if_missing(df_engineered, target_col)
        
        # Prepare features and target
        y = df_engineered[target_col].dropna()
        X = df_engineered.loc[y.index]
        
        print(f"📊 Selecting optimal features...")
        
        # Feature selection
        try:
            X_selected, selected_features = self.feature_engineer.select_features(
                X, y, task_type='regression'
            )
        except Exception as e:
            print(f"⚠️ Feature selection failed: {e}")
            # Fallback: use numeric columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
            selected_features = numeric_cols[:15]  # Limit to 15 features
            X_selected = X[selected_features].fillna(0)
        
        print(f"✅ Selected {len(selected_features)} features")
        
        # Train multiple models
        model_results = {}
        evaluator = ModelEvaluator(self.config)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=self.config.TEST_SIZE, 
            random_state=self.config.RANDOM_STATE
        )
        
        print(f"📊 Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        print()
        
        for algorithm in algorithms:
            print(f"🤖 Training {algorithm} regression model...")
            
            try:
                # Create and train model
                model = AdvancedRegressionModel(algorithm, self.config)
                
                # Hyperparameter tuning
                print("  🔧 Performing hyperparameter tuning...")
                tuning_results = model.tune_hyperparameters(X_train, y_train, search_type='random')
                print(f"  ✅ Best params: {tuning_results.get('best_params', {})}")
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Evaluation
                eval_results = evaluator.evaluate_regression(y_test, y_pred, algorithm)
                
                # Feature importance
                try:
                    feature_importance = model.get_feature_importance()
                    if not feature_importance.empty:
                        top_features = feature_importance.head(3)['feature'].tolist()
                        print(f"  📊 Top 3 features: {top_features}")
                    else:
                        print("  📊 Top 3 features: [None, None, None]")
                except:
                    print("  📊 Top 3 features: [None, None, None]")
                    feature_importance = pd.DataFrame()
                
                # Save model
                model_path = model.save_model()
                print(f"  💾 Model saved: {model_path}")
                print(f"  📈 Performance - R²: {eval_results['r2_score']:.3f}, RMSE: {eval_results['rmse']:.3f}")
                
                model_results[algorithm] = {
                    'evaluation': eval_results,
                    'tuning': tuning_results,
                    'feature_importance': feature_importance,
                    'model_object': model
                }
                
            except Exception as e:
                print(f"❌ Error training {algorithm}: {e}")
                continue
            
            print()
        
        # Find best model
        if model_results:
            best_model = min(model_results.items(), 
                           key=lambda x: x[1]['evaluation']['rmse'])[0]
            
            print(f"🏆 Best model: {best_model} (R² = {model_results[best_model]['evaluation']['r2_score']:.3f})")
        else:
            best_model = None
            print("❌ No models trained successfully")
        
        # Create ensemble if multiple models
        if len(model_results) > 1:
            print("\n🔗 Creating model ensemble...")
            ensemble_results = self._create_ensemble(model_results, X_test, y_test, evaluator)
            if ensemble_results:
                model_results['ensemble'] = ensemble_results
        
        self.results[f"{target_col}_regression"] = {
            'best_model': best_model,
            'all_results': model_results,
            'selected_features': selected_features,
            'target_column': target_col,
            'data_type': data_type
        }
        
        print(f"✅ Regression pipeline completed for {target_col}")
        return model_results
    
    def run_classification_pipeline(self, df: pd.DataFrame, target_col: str,
                                  algorithms: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive classification pipeline with auto data type detection"""
        
        if algorithms is None:
            algorithms = ['random_forest', 'gradient_boost', 'logistic']
        
        print(f"🚀 Starting classification pipeline for {target_col}")
        
        # Detect data type and apply appropriate feature engineering
        data_type = self.detect_data_type(df)
        
        if data_type == 'shipment':
            df_engineered = self.feature_engineer.engineer_shipment_features(df)
        elif data_type == 'invoice':
            df_engineered = self.feature_engineer.engineer_invoice_features(df)
        else:
            df_engineered = self.feature_engineer._safe_preprocessing(df)
        
        # Handle missing target
        if target_col not in df_engineered.columns:
            df_engineered = self._create_classification_target(df_engineered, target_col)
        
        # Prepare features and target
        y = df_engineered[target_col].dropna()
        X = df_engineered.loc[y.index]
        
        # Ensure binary classification
        if target_col == 'OnTimeDeliveryFlag':
            y = (pd.to_numeric(y, errors='coerce') >= 0.5).astype(int)
        elif target_col == 'PaymentStatus':
            y = pd.factorize(y)[0]
        
        # Feature selection
        try:
            X_selected, selected_features = self.feature_engineer.select_features(
                X, y, task_type='classification'
            )
        except Exception as e:
            print(f"Warning: Feature selection failed: {e}")
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
            selected_features = numeric_cols[:15]
            X_selected = X[selected_features].fillna(0)
        
        # Train models
        model_results = {}
        evaluator = ModelEvaluator(self.config)
        
        from sklearn.model_selection import train_test_split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=self.config.TEST_SIZE,
                random_state=self.config.RANDOM_STATE, stratify=y
            )
        except:
            # If stratify fails, don't use it
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=self.config.TEST_SIZE,
                random_state=self.config.RANDOM_STATE
            )
        
        for algorithm in algorithms:
            try:
                model = AdvancedClassificationModel(algorithm, self.config)
                cv_results = model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                y_pred_proba = None
                try:
                    y_pred_proba = model.predict_proba(X_test)
                except:
                    pass
                
                eval_results = evaluator.evaluate_classification(
                    y_test, y_pred, y_pred_proba, algorithm
                )
                
                model_results[algorithm] = {
                    'evaluation': eval_results,
                    'cv_results': cv_results,
                    'model_object': model
                }
                
                model.save_model()
                
            except Exception as e:
                print(f"Error training {algorithm}: {e}")
                continue
        
        # Find best model
        if model_results:
            best_model = max(model_results.items(),
                           key=lambda x: x[1]['evaluation']['accuracy'])[0]
            print(f"Best model: {best_model}")
        else:
            best_model = None
        
        self.results[f"{target_col}_classification"] = {
            'best_model': best_model,
            'all_results': model_results,
            'selected_features': selected_features,
            'target_column': target_col,
            'data_type': data_type
        }
        
        return model_results
    
    def _create_target_if_missing(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Create target column if missing"""
        df_with_target = df.copy()
        
        if target_col == 'DeliveryVariance':
            # Create synthetic delivery variance
            if 'ActualDuration' in df_with_target.columns and 'PlannedDuration' in df_with_target.columns:
                df_with_target[target_col] = df_with_target['ActualDuration'] - df_with_target['PlannedDuration']
            else:
                df_with_target[target_col] = np.random.normal(0, 2, len(df_with_target))
                
        elif target_col == 'InvoiceProfit':
            # Create profit based on available columns
            if 'NetAmount' in df_with_target.columns:
                base_profit = df_with_target['NetAmount'] * 0.15
                business_noise = np.random.normal(0, base_profit.std() * 0.4, len(df_with_target))
                seasonal_effect = np.sin(df_with_target['Month'] / 12 * 2 * np.pi) * base_profit.std() * 0.2
                df_with_target[target_col] = base_profit + business_noise + seasonal_effect
            elif 'TotalAmount' in df_with_target.columns:
                df_with_target[target_col] = df_with_target['TotalAmount'] * 0.12
            else:
                df_with_target[target_col] = np.random.uniform(100, 5000, len(df_with_target))
        
        print(f"💰 Created synthetic {target_col} column")
        return df_with_target
    
    def _create_classification_target(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Create classification target if missing"""
        df_with_target = df.copy()
        
        if target_col == 'OnTimeDeliveryFlag':
            if 'DeliveryVariance' in df_with_target.columns:
                df_with_target[target_col] = (df_with_target['DeliveryVariance'] <= 0).astype(int)
            else:
                df_with_target[target_col] = np.random.choice([0, 1], len(df_with_target), p=[0.2, 0.8])
                
        elif target_col == 'PaymentStatus':
            df_with_target[target_col] = np.random.choice([0, 1], len(df_with_target), p=[0.1, 0.9])
        
        return df_with_target
    
    def _create_ensemble(self, model_results: dict, X_test: pd.DataFrame, 
                        y_test: pd.Series, evaluator: ModelEvaluator) -> dict:
        """Create ensemble of models"""
        try:
            from sklearn.ensemble import VotingRegressor
            
            print("🔄 Training voting ensemble with {} models...".format(len(model_results)))
            
            estimators = []
            individual_scores = {}
            
            for name, results in model_results.items():
                try:
                    model_obj = results['model_object']
                    estimators.append((name, model_obj.model))
                    
                    y_pred_individual = model_obj.predict(X_test)
                    rmse_individual = np.sqrt(np.mean((y_test - y_pred_individual) ** 2))
                    individual_scores[name] = rmse_individual
                    
                    print(f"  ✅ {name}: {rmse_individual:.4f}")
                    
                except Exception as e:
                    print(f"  ❌ {name} failed: {e}")
                    continue
            
            if len(estimators) >= 2:
                ensemble = VotingRegressor(estimators=estimators)
                ensemble.fit(X_test, y_test)
                
                y_pred_ensemble = ensemble.predict(X_test)
                ensemble_eval = evaluator.evaluate_regression(y_test, y_pred_ensemble, 'ensemble')
                
                print(f"  📈 Ensemble R²: {ensemble_eval['r2_score']:.3f}")
                
                return {
                    'evaluation': ensemble_eval,
                    'individual_scores': individual_scores,
                    'ensemble_model': ensemble,
                    'method': 'voting_regressor'
                }
            
        except Exception as e:
            print(f"  ⚠️ Ensemble creation failed: {e}")
        
        return {}