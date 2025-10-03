# ML/models/time_series/advanced_forecasting.py

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import timedelta
import warnings
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from ML.config.model_config import ModelConfig

warnings.filterwarnings('ignore')

class TimeSeriesForecaster:
    """Advanced time series forecasting with multiple methods"""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.models = {}
        self.fitted_models = {}
    
    def prepare_time_series(self, df: pd.DataFrame, date_col: str, 
                          value_col: str, freq: str = 'D') -> pd.Series:
        """Prepare time series data"""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col, value_col])
        df = df.set_index(date_col).sort_index()
        
        # Resample to ensure regular frequency
        ts = df[value_col].resample(freq).sum()
        ts = ts.fillna(0)  # Fill missing periods with 0
        
        return ts
    
    def exponential_smoothing_forecast(self, series: pd.Series, 
                                     periods: int) -> Tuple[pd.Series, Dict]:
        """Exponential smoothing forecast with error handling"""
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            # Determine seasonality
            seasonal_periods = min(12, len(series) // 3) if len(series) > 24 else None
            
            if seasonal_periods and seasonal_periods > 1:
                model = ExponentialSmoothing(
                    series, 
                    trend='add', 
                    seasonal='add',
                    seasonal_periods=seasonal_periods
                )
            else:
                model = ExponentialSmoothing(series, trend='add')
            
            fitted_model = model.fit(optimized=True)
            forecast = fitted_model.forecast(periods)
            
            # Create forecast index
            last_date = series.index[-1]
            forecast_index = pd.date_range(
                start=last_date + timedelta(days=1), 
                periods=periods, 
                freq='D'
            )
            forecast_series = pd.Series(forecast, index=forecast_index)
            
            metrics = {
                'aic': fitted_model.aic,
                'method': 'exponential_smoothing'
            }
            
            return forecast_series, metrics
            
        except Exception as e:
            print(f"⚠️ Exponential smoothing failed: {e}, using fallback")
            return self._fallback_forecast(series, periods)
    
    def arima_forecast(self, series: pd.Series, periods: int) -> Tuple[pd.Series, Dict]:
        """ARIMA forecast with auto parameter selection"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.stattools import adfuller
            
            # Check stationarity
            adf_result = adfuller(series.dropna())
            is_stationary = adf_result[1] < 0.05
            
            # Simple ARIMA parameter selection
            best_aic = np.inf
            best_order = (1, 1, 1)
            
            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        try:
                            model = ARIMA(series, order=(p, d, q))
                            fitted = model.fit()
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_order = (p, d, q)
                        except:
                            continue
            
            # Fit best model and forecast
            model = ARIMA(series, order=best_order)
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=periods)
            
            # Create forecast index
            last_date = series.index[-1]
            forecast_index = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=periods,
                freq='D'
            )
            forecast_series = pd.Series(forecast, index=forecast_index)
            
            metrics = {
                'aic': fitted_model.aic,
                'order': best_order,
                'method': 'arima'
            }
            
            return forecast_series, metrics
            
        except Exception as e:
            print(f"⚠️ ARIMA failed: {e}, using fallback")
            return self._fallback_forecast(series, periods)
    
    def prophet_forecast(self, series: pd.Series, periods: int) -> Tuple[pd.Series, Dict]:
        """Facebook Prophet forecast (if available)"""
        try:
            from prophet import Prophet
            
            # Prepare data for Prophet
            df = pd.DataFrame({
                'ds': series.index,
                'y': series.values
            })
            
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            model.fit(df)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            
            # Extract forecast values
            forecast_values = forecast['yhat'].tail(periods)
            forecast_index = pd.date_range(
                start=series.index[-1] + timedelta(days=1),
                periods=periods,
                freq='D'
            )
            forecast_series = pd.Series(forecast_values.values, index=forecast_index)
            
            metrics = {
                'method': 'prophet',
                'components': ['trend', 'weekly', 'yearly']
            }
            
            return forecast_series, metrics
            
        except ImportError:
            print("⚠️ Prophet not available, using exponential smoothing")
            return self.exponential_smoothing_forecast(series, periods)
        except Exception as e:
            print(f"⚠️ Prophet failed: {e}, using fallback")
            return self._fallback_forecast(series, periods)
    
    def _fallback_forecast(self, series: pd.Series, periods: int) -> Tuple[pd.Series, Dict]:
        """Fallback forecasting method"""
        if len(series) >= 7:
            # Use seasonal naive (last week pattern)
            last_week = series.tail(7)
            pattern = last_week.values
            forecast_values = np.tile(pattern, (periods // 7) + 1)[:periods]
        else:
            # Use simple mean
            forecast_values = [series.mean()] * periods
        
        forecast_index = pd.date_range(
            start=series.index[-1] + timedelta(days=1),
            periods=periods,
            freq='D'
        )
        forecast_series = pd.Series(forecast_values, index=forecast_index)
        
        metrics = {
            'method': 'seasonal_naive' if len(series) >= 7 else 'mean',
            'warning': 'fallback_method_used'
        }
        
        return forecast_series, metrics
    
    def ensemble_forecast(self, series: pd.Series, periods: int) -> Tuple[pd.Series, Dict]:
        """Ensemble forecast combining multiple methods"""
        forecasts = {}
        methods = ['exponential_smoothing', 'arima']
        
        # Try Prophet if available
        try:
            import prophet
            methods.append('prophet')
        except ImportError:
            pass
        
        # Generate forecasts from different methods
        for method in methods:
            try:
                if method == 'exponential_smoothing':
                    forecast, _ = self.exponential_smoothing_forecast(series, periods)
                elif method == 'arima':
                    forecast, _ = self.arima_forecast(series, periods)
                elif method == 'prophet':
                    forecast, _ = self.prophet_forecast(series, periods)
                
                forecasts[method] = forecast
            except Exception as e:
                print(f"⚠️ {method} failed in ensemble: {e}")
        
        if not forecasts:
            return self._fallback_forecast(series, periods)
        
        # Simple equal weighting ensemble
        ensemble_values = np.mean([f.values for f in forecasts.values()], axis=0)
        
        forecast_index = pd.date_range(
            start=series.index[-1] + timedelta(days=1),
            periods=periods,
            freq='D'
        )
        ensemble_series = pd.Series(ensemble_values, index=forecast_index)
        
        metrics = {
            'method': 'ensemble',
            'components': list(forecasts.keys()),
            'weights': 'equal'
        }
        
        return ensemble_series, metrics
    
    def analyze_seasonality(self, series: pd.Series) -> Dict:
        """Analyze seasonality patterns"""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            if len(series) >= 24:  # Need enough data for decomposition
                decomposition = seasonal_decompose(series, model='additive', period=7)
                
                return {
                    'has_seasonality': True,
                    'seasonal_strength': np.var(decomposition.seasonal) / np.var(series),
                    'trend_strength': np.var(decomposition.trend.dropna()) / np.var(series),
                    'residual_variance': np.var(decomposition.resid.dropna())
                }
            else:
                return {'has_seasonality': False, 'reason': 'insufficient_data'}
                
        except Exception as e:
            return {'has_seasonality': False, 'error': str(e)}
    
    def analyze_trend(self, series: pd.Series) -> Dict:
        """Analyze trend patterns"""
        try:
            from scipy import stats
            
            # Linear trend
            x = np.arange(len(series))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, series.values)
            
            # Trend classification
            if abs(slope) < 0.01:
                trend_type = 'stationary'
            elif slope > 0:
                trend_type = 'increasing'
            else:
                trend_type = 'decreasing'
            
            return {
                'trend_type': trend_type,
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'is_significant': p_value < 0.05
            }
            
        except Exception as e:
            return {'trend_type': 'unknown', 'error': str(e)}