# ML/pipelines/prediction_pipeline.py

import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, List, Optional
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ML.config.model_config import ModelConfig
from ML.features.engineering import FeatureEngineer

class PredictionPipeline:
    """Production prediction pipeline"""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.loaded_models = {}
        self.feature_engineer = FeatureEngineer()
        self.model_metadata = {}
    
    def load_production_models(self) -> Dict[str, bool]:
        """Load all production models"""
        
        if not os.path.exists(self.config.MODEL_DIR):
            print(f"❌ Model directory not found: {self.config.MODEL_DIR}")
            return {}
        
        model_files = [f for f in os.listdir(self.config.MODEL_DIR) if f.endswith('.joblib')]
        
        if not model_files:
            print("⚠️ No model files found")
            return {}
        
        results = {}
        for model_file in model_files:
            model_name = model_file.replace('.joblib', '')
            try:
                model_path = os.path.join(self.config.MODEL_DIR, model_file)
                model_data = joblib.load(model_path)
                
                # Validate model data structure
                required_keys = ['model', 'feature_names']
                if all(key in model_data for key in required_keys):
                    self.loaded_models[model_name] = model_data
                    self.model_metadata[model_name] = {
                        'loaded_at': datetime.now(),
                        'feature_count': len(model_data['feature_names']),
                        'model_type': type(model_data['model']).__name__
                    }
                    results[model_name] = True
                    print(f"✅ Loaded model: {model_name}")
                else:
                    print(f"⚠️ Invalid model data structure: {model_name}")
                    results[model_name] = False
                    
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                results[model_name] = False
        
        print(f"📊 Successfully loaded {sum(results.values())}/{len(results)} models")
        return results
    
    def predict_delivery_performance(self, shipment_data: pd.DataFrame) -> pd.DataFrame:
        """Predict delivery performance for new shipments"""
        
        print("🚛 Predicting delivery performance...")
        
        # Feature engineering
        df_features = self.feature_engineer.engineer_shipment_features(shipment_data)
        results = shipment_data.copy()
        
        # On-time delivery prediction
        ontime_models = [m for m in self.loaded_models.keys() if 'ontime' in m.lower() or 'classification' in m]
        
        if ontime_models:
            model_name = ontime_models[0]  # Use first available
            model_data = self.loaded_models[model_name]
            model = model_data['model']
            feature_names = model_data['feature_names']
            
            try:
                # Prepare features
                X = df_features[feature_names].fillna(0)
                
                # Predictions
                ontime_pred = model.predict(X)
                results['PredictedOnTimeFlag'] = ontime_pred
                
                # Probabilities if available
                if hasattr(model, 'predict_proba'):
                    ontime_proba = model.predict_proba(X)
                    if ontime_proba.shape[1] > 1:
                        results['OnTimeProbability'] = ontime_proba[:, 1]
                    else:
                        results['OnTimeProbability'] = ontime_proba[:, 0]
                else:
                    results['OnTimeProbability'] = ontime_pred.astype(float)
                
                print(f"  ✅ On-time predictions: {ontime_pred.mean():.1%} predicted on-time")
                
            except Exception as e:
                print(f"  ❌ Error in on-time prediction: {e}")
                results['PredictedOnTimeFlag'] = 0
                results['OnTimeProbability'] = 0.5
        
        # Delivery variance prediction
        variance_models = [m for m in self.loaded_models.keys() if 'variance' in m.lower() or 'regression' in m]
        
        if variance_models:
            model_name = variance_models[0]  # Use first available
            model_data = self.loaded_models[model_name]
            model = model_data['model']
            feature_names = model_data['feature_names']
            
            try:
                # Prepare features
                available_features = [f for f in feature_names if f in df_features.columns]
                if len(available_features) < len(feature_names) * 0.7:  # Less than 70% features available
                    print(f"  ⚠️ Only {len(available_features)}/{len(feature_names)} features available")
                
                X = df_features[available_features].fillna(0)
                
                # Add missing features as zeros
                for missing_feat in set(feature_names) - set(available_features):
                    X[missing_feat] = 0
                
                # Reorder columns to match training
                X = X[feature_names]
                
                variance_pred = model.predict(X)
                results['PredictedDeliveryVariance'] = variance_pred
                
                # Calculate predicted arrival date
                if 'PlannedArrivalDate' in results.columns:
                    results['PredictedArrivalDate'] = (
                        pd.to_datetime(results['PlannedArrivalDate']) + 
                        pd.to_timedelta(variance_pred, unit='D')
                    )
                
                print(f"  ✅ Variance predictions: avg {variance_pred.mean():.1f} days")
                
            except Exception as e:
                print(f"  ❌ Error in variance prediction: {e}")
                results['PredictedDeliveryVariance'] = 0
        
        return results
    
    def predict_invoice_payment(self, invoice_data: pd.DataFrame) -> pd.DataFrame:
        """Predict invoice payment status and timing"""
        
        print("💰 Predicting invoice payment status...")
        
        # Feature engineering
        df_features = self.feature_engineer.engineer_invoice_features(invoice_data)
        results = invoice_data.copy()
        
        # Payment status prediction
        payment_models = [m for m in self.loaded_models.keys() if 'payment' in m.lower()]
        
        if payment_models:
            model_name = payment_models[0]
            model_data = self.loaded_models[model_name]
            model = model_data['model']
            feature_names = model_data['feature_names']
            
            try:
                # Prepare features
                available_features = [f for f in feature_names if f in df_features.columns]
                X = df_features[available_features].fillna(0)
                
                # Add missing features as zeros
                for missing_feat in set(feature_names) - set(available_features):
                    X[missing_feat] = 0
                
                X = X[feature_names]
                
                payment_pred = model.predict(X)
                results['PredictedPaymentStatus'] = payment_pred
                
                # Payment probabilities
                if hasattr(model, 'predict_proba'):
                    payment_proba = model.predict_proba(X)
                    results['PaymentProbability'] = np.max(payment_proba, axis=1)
                
                print(f"  ✅ Payment predictions completed")
                
            except Exception as e:
                print(f"  ❌ Error in payment prediction: {e}")
                results['PredictedPaymentStatus'] = 1  # Default to paid
                results['PaymentProbability'] = 0.5
        
        return results
    
    def batch_predict(self, data: pd.DataFrame, model_name: str, 
                     prediction_type: str = 'auto') -> np.ndarray:
        """Generic batch prediction for any loaded model"""
        
        if model_name not in self.loaded_models:
            available_models = list(self.loaded_models.keys())
            raise ValueError(f"Model {model_name} not loaded. Available: {available_models}")
        
        model_data = self.loaded_models[model_name]
        model = model_data['model']
        feature_names = model_data['feature_names']
        
        print(f"🔮 Making batch predictions with {model_name}...")
        
        # Prepare features based on data type
        if prediction_type == 'auto':
            # Auto-detect based on columns
            if 'ShipmentDate' in data.columns:
                data_features = self.feature_engineer.engineer_shipment_features(data)
            else:
                data_features = self.feature_engineer.engineer_invoice_features(data)
        elif prediction_type == 'shipment':
            data_features = self.feature_engineer.engineer_shipment_features(data)
        elif prediction_type == 'invoice':
            data_features = self.feature_engineer.engineer_invoice_features(data)
        else:
            data_features = data.copy()
        
        # Prepare feature matrix
        available_features = [f for f in feature_names if f in data_features.columns]
        missing_features = set(feature_names) - set(available_features)
        
        if missing_features:
            print(f"⚠️ Missing features: {list(missing_features)[:5]}... (setting to 0)")
        
        X = data_features[available_features].fillna(0)
        
        # Add missing features as zeros
        for missing_feat in missing_features:
            X[missing_feat] = 0
        
        # Reorder columns to match training
        X = X[feature_names]
        
        # Make predictions
        predictions = model.predict(X)
        
        print(f"✅ Generated {len(predictions)} predictions")
        return predictions
    
    def predict_with_confidence(self, data: pd.DataFrame, model_name: str,
                              confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        """Make predictions with confidence intervals"""
        
        if model_name not in self.loaded_models:
            raise ValueError(f"Model {model_name} not loaded")
        
        # Get base predictions
        predictions = self.batch_predict(data, model_name)
        
        # For ensemble models, we can estimate uncertainty
        # For single models, we'll use bootstrap sampling approximation
        
        try:
            # Simple confidence estimation using historical performance
            model_data = self.loaded_models[model_name]
            
            if 'performance_history' in model_data:
                performance = model_data['performance_history']
                if performance:
                    # Use CV standard deviation as uncertainty estimate
                    cv_std = performance[-1].get('cv_std', 0.1)
                    
                    # Calculate confidence intervals
                    from scipy import stats
                    alpha = 1 - confidence_level
                    z_score = stats.norm.ppf(1 - alpha/2)
                    
                    margin_of_error = z_score * cv_std * np.ones_like(predictions)
                    
                    lower_bound = predictions - margin_of_error
                    upper_bound = predictions + margin_of_error
                    
                    return {
                        'predictions': predictions,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'confidence_level': confidence_level
                    }
        
        except Exception as e:
            print(f"⚠️ Could not calculate confidence intervals: {e}")
        
        # Fallback: return predictions without confidence intervals
        return {
            'predictions': predictions,
            'lower_bound': predictions,
            'upper_bound': predictions,
            'confidence_level': confidence_level
        }
    
    def get_model_info(self, model_name: str = None) -> Dict[str, Any]:
        """Get information about loaded models"""
        
        if model_name:
            if model_name not in self.loaded_models:
                return {}
            
            model_data = self.loaded_models[model_name]
            metadata = self.model_metadata.get(model_name, {})
            
            return {
                'model_name': model_name,
                'model_type': metadata.get('model_type', 'unknown'),
                'feature_count': metadata.get('feature_count', 0),
                'loaded_at': metadata.get('loaded_at'),
                'feature_names': model_data.get('feature_names', []),
                'performance_history': model_data.get('performance_history', [])
            }
        else:
            # Return info for all models
            all_info = {}
            for name in self.loaded_models.keys():
                all_info[name] = self.get_model_info(name)
            return all_info
    
    def validate_input_data(self, data: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """Validate input data for predictions"""
        
        if model_name not in self.loaded_models:
            return {'valid': False, 'error': f'Model {model_name} not loaded'}
        
        model_data = self.loaded_models[model_name]
        required_features = model_data['feature_names']
        
        validation_result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'data_info': {
                'rows': len(data),
                'columns': len(data.columns),
                'missing_features': [],
                'extra_features': list(set(data.columns) - set(required_features))
            }
        }
        
        # Check for completely missing required features
        missing_features = set(required_features) - set(data.columns)
        if missing_features:
            validation_result['warnings'].append(f"Missing features will be set to 0: {list(missing_features)}")
            validation_result['data_info']['missing_features'] = list(missing_features)
        
        # Check data types
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        non_numeric_required = set(required_features) - set(numeric_cols)
        if non_numeric_required:
            validation_result['warnings'].append(f"Non-numeric features detected: {list(non_numeric_required)}")
        
        # Check for missing values
        missing_values = data[data.columns.intersection(required_features)].isnull().sum()
        if missing_values.any():
            validation_result['warnings'].append(f"Missing values detected in: {missing_values[missing_values > 0].to_dict()}")
        
        # Check data ranges (basic sanity check)
        for col in data.select_dtypes(include=[np.number]).columns:
            if col in required_features:
                if (data[col] < 0).any() and col in ['TotalWeight', 'TotalVolume', 'ShipmentValue']:
                    validation_result['warnings'].append(f"Negative values in {col} (unusual for logistics data)")
        
        return validation_result
    
    def predict_shipment_risks(self, shipment_data: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive shipment risk assessment"""
        
        print("⚠️ Assessing shipment risks...")
        
        results = shipment_data.copy()
        risk_scores = pd.DataFrame(index=shipment_data.index)
        
        # Delivery delay risk
        try:
            delivery_pred = self.predict_delivery_performance(shipment_data)
            if 'OnTimeProbability' in delivery_pred.columns:
                risk_scores['DelayRisk'] = 1 - delivery_pred['OnTimeProbability']
            else:
                risk_scores['DelayRisk'] = 0.5  # Default medium risk
        except Exception as e:
            print(f"⚠️ Could not assess delay risk: {e}")
            risk_scores['DelayRisk'] = 0.5
        
        # Weight/volume risk assessment
        if all(col in shipment_data.columns for col in ['TotalWeight', 'TotalVolume']):
            # Calculate percentiles for risk assessment
            weight_percentile = shipment_data['TotalWeight'].rank(pct=True)
            volume_percentile = shipment_data['TotalVolume'].rank(pct=True)
            
            # High weight/volume = higher risk
            risk_scores['WeightRisk'] = weight_percentile
            risk_scores['VolumeRisk'] = volume_percentile
        
        # Value risk assessment
        if 'ShipmentValue' in shipment_data.columns:
            value_percentile = shipment_data['ShipmentValue'].rank(pct=True)
            risk_scores['ValueRisk'] = value_percentile  # High value = higher risk
        
        # Route complexity risk
        if all(col in shipment_data.columns for col in ['OriginLocationKey', 'DestinationLocationKey']):
            route_complexity = abs(shipment_data['OriginLocationKey'] - shipment_data['DestinationLocationKey'])
            complexity_percentile = route_complexity.rank(pct=True)
            risk_scores['RouteRisk'] = complexity_percentile
        
        # Calculate overall risk score
        risk_columns = [col for col in risk_scores.columns if 'Risk' in col]
        if risk_columns:
            risk_scores['OverallRisk'] = risk_scores[risk_columns].mean(axis=1)
            
            # Risk categories
            risk_scores['RiskCategory'] = pd.cut(
                risk_scores['OverallRisk'],
                bins=[0, 0.3, 0.7, 1.0],
                labels=['Low', 'Medium', 'High']
            )
        
        # Add risk scores to results
        for col in risk_scores.columns:
            results[col] = risk_scores[col]
        
        # Risk summary
        if 'RiskCategory' in results.columns:
            risk_summary = results['RiskCategory'].value_counts()
            print(f"  📊 Risk distribution: {dict(risk_summary)}")
        
        return results
    
    def batch_forecast(self, historical_data: pd.DataFrame, 
                      date_col: str, value_col: str,
                      horizon: int = 30) -> Dict[str, Any]:
        """Generate batch forecasts for time series data"""
        
        print(f"📈 Generating {horizon}-day forecast for {value_col}...")
        
        # Import forecaster
        from ML.models.time_series.advanced_forecasting import TimeSeriesForecaster
        
        forecaster = TimeSeriesForecaster(self.config)
        
        # Prepare time series
        ts = forecaster.prepare_time_series(historical_data, date_col, value_col)
        
        if len(ts) < 7:
            print("⚠️ Insufficient historical data for forecasting")
            return None
        
        # Generate ensemble forecast
        forecast, metrics = forecaster.ensemble_forecast(ts, horizon)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Date': forecast.index,
            'PredictedValue': forecast.values,
            'ForecastMethod': metrics['method'],
            'IsWeekend': forecast.index.weekday.isin([5, 6])
        })
        
        # Add confidence intervals (simplified)
        historical_std = ts.std()
        forecast_df['LowerBound'] = forecast_df['PredictedValue'] - 1.96 * historical_std
        forecast_df['UpperBound'] = forecast_df['PredictedValue'] + 1.96 * historical_std
        
        # Save forecast
        forecast_path = os.path.join(self.config.OUTPUT_DIR, f"forecast_{value_col}_{datetime.now().strftime('%Y%m%d')}.csv")
        forecast_df.to_csv(forecast_path, index=False)
        
        return {
            'forecast_data': forecast_df,
            'historical_data': ts,
            'metrics': metrics,
            'forecast_path': forecast_path
        }
    
    def explain_prediction(self, data_point: pd.Series, model_name: str) -> Dict[str, Any]:
        """Explain individual prediction (simplified SHAP-like explanation)"""
        
        if model_name not in self.loaded_models:
            return {'error': f'Model {model_name} not available'}
        
        model_data = self.loaded_models[model_name]
        model = model_data['model']
        feature_names = model_data['feature_names']
        
        try:
            # Prepare single data point
            if 'ShipmentDate' in data_point.index:
                data_features = self.feature_engineer.engineer_shipment_features(pd.DataFrame([data_point]))
            else:
                data_features = self.feature_engineer.engineer_invoice_features(pd.DataFrame([data_point]))
            
            # Get feature values
            feature_values = {}
            for feat in feature_names:
                if feat in data_features.columns:
                    feature_values[feat] = data_features[feat].iloc[0]
                else:
                    feature_values[feat] = 0
            
            X = pd.DataFrame([feature_values])
            prediction = model.predict(X)[0]
            
            # Simple feature contribution analysis
            # For tree-based models, we can use feature importance as proxy
            feature_contributions = {}
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                for i, feat in enumerate(feature_names):
                    # Contribution = feature_value * importance * sign_of_contribution
                    feature_contributions[feat] = {
                        'value': feature_values[feat],
                        'importance': importances[i],
                        'contribution_score': feature_values[feat] * importances[i]
                    }
            
            # Sort by contribution
            sorted_contributions = sorted(
                feature_contributions.items(),
                key=lambda x: abs(x[1]['contribution_score']),
                reverse=True
            )
            
            return {
                'prediction': prediction,
                'feature_contributions': dict(sorted_contributions[:10]),  # Top 10
                'model_used': model_name,
                'explanation_method': 'feature_importance_weighted'
            }
            
        except Exception as e:
            return {'error': f'Explanation failed: {e}'}
    
    def get_prediction_summary(self) -> Dict[str, Any]:
        """Get summary of prediction pipeline status"""
        
        return {
            'loaded_models': len(self.loaded_models),
            'model_list': list(self.loaded_models.keys()),
            'model_metadata': self.model_metadata,
            'available_prediction_types': [
                'delivery_performance',
                'invoice_payment', 
                'batch_predict',
                'shipment_risks',
                'batch_forecast'
            ],
            'last_updated': datetime.now().isoformat()
        }

