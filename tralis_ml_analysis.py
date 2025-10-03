# ================================
# ML/config/model_config.py
# ================================

import os
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class ModelConfig:
    """Configuration for ML models and pipelines"""
    
    # Directories
    BASE_DIR: str = os.path.join(os.getcwd(), "ML")
    MODEL_DIR: str = os.path.join(BASE_DIR, "saved_models")
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "outputs")
    PLOTS_DIR: str = os.path.join(OUTPUT_DIR, "plots")
    REPORTS_DIR: str = os.path.join(OUTPUT_DIR, "reports")
    
    # Model parameters
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    CV_FOLDS: int = 5
    
    # Time series
    TS_HORIZON_DAYS: int = 30
    TS_HORIZON_MONTHS: int = 6
    TS_SEASONAL_PERIODS: int = 7
    
    # Customer segmentation
    N_CLUSTERS: int = 4
    RFM_QUANTILES: int = 4
    
    # Performance thresholds
    MIN_ACCURACY: float = 0.70
    MIN_R2: float = 0.60
    MAX_MAPE: float = 20.0
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        for dir_path in [self.MODEL_DIR, self.OUTPUT_DIR, self.PLOTS_DIR, self.REPORTS_DIR]:
            os.makedirs(dir_path, exist_ok=True)

# ================================
# ML/config/feature_config.py
# ================================

class FeatureConfig:
    """Feature engineering configuration"""
    
    SHIPMENT_FEATURES = {
        'base': ['TotalWeight', 'TotalVolume', 'TotalPackages', 'ShipmentValue'],
        'temporal': ['Month', 'DayOfWeek', 'Quarter', 'Year'],
        'categorical': ['CustomerKey', 'OriginLocationKey', 'DestinationLocationKey', 
                       'EquipmentKey', 'FreightTypeKey'],
        'engineered': ['WeightVolumeDensity', 'ValuePerWeight', 'ValuePerPackage']
    }
    
    INVOICE_FEATURES = {
        'base': ['TotalAmount', 'NetAmount', 'TaxAmount'],
        'temporal': ['Month', 'Quarter', 'DayOfWeek', 'Year'],
        'categorical': ['CustomerKey', 'SupplierKey', 'PaymentTermKey'],
        'engineered': ['TaxRate', 'AmountPerDay', 'ProfitMargin']
    }
    
    SCALING_METHODS = {
        'robust': 'RobustScaler',
        'standard': 'StandardScaler', 
        'minmax': 'MinMaxScaler'
    }

# ================================
# ML/data/loaders.py
# ================================

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from ..config.model_config import ModelConfig
from Configuration.db_config import get_target_engine

config = ModelConfig()

class DataLoader:
    """Enhanced data loading with caching and validation"""
    
    def __init__(self):
        self.engine = get_target_engine()
        self._cache = {}
    
    def load_shipments(self, limit: Optional[int] = None, use_cache: bool = True) -> pd.DataFrame:
        """Load shipments with enhanced error handling and caching"""
        cache_key = f"shipments_{limit}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        try:
            query = "SELECT * FROM FactShipments"
            if limit:
                query = f"SELECT TOP {limit} * FROM FactShipments"
            
            df = pd.read_sql(query, self.engine)
            
            # Basic validation
            if df.empty:
                raise ValueError("No shipment data found")
            
            # Cache result
            if use_cache:
                self._cache[cache_key] = df.copy()
                
            print(f"✅ Loaded {len(df)} shipment records")
            return df
            
        except Exception as e:
            print(f"❌ Error loading shipments: {e}")
            return pd.DataFrame()
    
    def load_invoices(self, limit: Optional[int] = None, use_cache: bool = True) -> pd.DataFrame:
        """Load invoices with enhanced error handling and caching"""
        cache_key = f"invoices_{limit}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        try:
            query = "SELECT * FROM FactInvoices"
            if limit:
                query = f"SELECT TOP {limit} * FROM FactInvoices"
            
            df = pd.read_sql(query, self.engine)
            
            if df.empty:
                raise ValueError("No invoice data found")
            
            if use_cache:
                self._cache[cache_key] = df.copy()
                
            print(f"✅ Loaded {len(df)} invoice records")
            return df
            
        except Exception as e:
            print(f"❌ Error loading invoices: {e}")
            return pd.DataFrame()
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of loaded data"""
        summary = {}
        
        for key, df in self._cache.items():
            summary[key] = {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
                'date_range': self._get_date_range(df)
            }
        
        return summary
    
    def _get_date_range(self, df: pd.DataFrame) -> str:
        """Extract date range from dataframe"""
        date_cols = [col for col in df.columns if 'Date' in col]
        if not date_cols:
            return "No date columns"
        
        try:
            dates = pd.to_datetime(df[date_cols[0]], errors='coerce').dropna()
            if dates.empty:
                return "No valid dates"
            return f"{dates.min().date()} to {dates.max().date()}"
        except:
            return "Date parsing error"

# ================================
# ML/features/engineering.py
# ================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from typing import List, Tuple, Dict, Any

class FeatureEngineer:
    """Advanced feature engineering for logistics data"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = {}
        
    def engineer_shipment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for shipment data"""
        df = df.copy()
        
        # Density and efficiency metrics
        df['WeightVolumeDensity'] = df['TotalWeight'] / (df['TotalVolume'] + 1e-6)
        df['ValuePerWeight'] = df['ShipmentValue'] / (df['TotalWeight'] + 1e-6)
        df['ValuePerPackage'] = df['ShipmentValue'] / (df['TotalPackages'] + 1e-6)
        df['PackageDensity'] = df['TotalPackages'] / (df['TotalVolume'] + 1e-6)
        
        # Time-based features
        if 'ShipmentDate' in df.columns:
            df['ShipmentDate'] = pd.to_datetime(df['ShipmentDate'], errors='coerce')
            df['Month'] = df['ShipmentDate'].dt.month
            df['Quarter'] = df['ShipmentDate'].dt.quarter
            df['DayOfWeek'] = df['ShipmentDate'].dt.dayofweek
            df['Year'] = df['ShipmentDate'].dt.year
            df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
            df['DayOfMonth'] = df['ShipmentDate'].dt.day
            df['WeekOfYear'] = df['ShipmentDate'].dt.isocalendar().week
            
            # Cyclical encoding for temporal features
            df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
            df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
            df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
            df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        
        # Delivery performance features
        if all(col in df.columns for col in ['PlannedArrivalDate', 'ActualArrivalDate']):
            df['PlannedDuration'] = (df['PlannedArrivalDate'] - df['ShipmentDate']).dt.days
            df['ActualDuration'] = (df['ActualArrivalDate'] - df['ShipmentDate']).dt.days
            df['DeliveryVariance'] = df['ActualDuration'] - df['PlannedDuration']
            df['OnTimeDeliveryFlag'] = (df['DeliveryVariance'] <= 0).astype(int)
        
        # Business intelligence features
        df['ShipmentSize'] = pd.cut(df['TotalWeight'], bins=3, labels=['Small', 'Medium', 'Large'])
        df['ValueCategory'] = pd.cut(df['ShipmentValue'], bins=4, labels=['Low', 'Medium', 'High', 'Premium'])
        
        return df
    
    def engineer_invoice_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for invoice data"""
        df = df.copy()
        
        # Financial metrics
        df['TaxRate'] = df['TaxAmount'] / (df['NetAmount'] + 1e-6)
        df['ProfitMargin'] = (df['NetAmount'] - df['TotalAmount']) / (df['NetAmount'] + 1e-6)
        
        # Time-based features
        if 'InvoiceDate' in df.columns:
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
            df['Month'] = df['InvoiceDate'].dt.month
            df['Quarter'] = df['InvoiceDate'].dt.quarter
            df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
            df['Year'] = df['InvoiceDate'].dt.year
            df['IsMonthEnd'] = (df['InvoiceDate'].dt.day > 25).astype(int)
            df['IsQuarterEnd'] = df['InvoiceDate'].dt.month.isin([3, 6, 9, 12]).astype(int)
            
            # Cyclical encoding
            df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
            df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        # Payment features
        if all(col in df.columns for col in ['InvoiceDate', 'PaymentDueDate']):
            df['PaymentPeriod'] = (df['PaymentDueDate'] - df['InvoiceDate']).dt.days
            df['IsOverdue'] = (pd.Timestamp.now() > df['PaymentDueDate']).astype(int)
        
        # Amount categories
        df['AmountCategory'] = pd.cut(df['NetAmount'], bins=5, labels=['XS', 'S', 'M', 'L', 'XL'])
        
        return df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       task_type: str = 'regression', k: int = 15) -> Tuple[pd.DataFrame, List[str]]:
        """Intelligent feature selection based on statistical tests"""
        
        # Separate numeric and categorical features
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Encode categorical features
        X_encoded = X.copy()
        for cat_col in categorical_features:
            if cat_col not in self.encoders:
                self.encoders[cat_col] = LabelEncoder()
                X_encoded[cat_col] = self.encoders[cat_col].fit_transform(X_encoded[cat_col].astype(str))
            else:
                X_encoded[cat_col] = self.encoders[cat_col].transform(X_encoded[cat_col].astype(str))
        
        # Feature selection
        if task_type == 'regression':
            selector = SelectKBest(score_func=f_regression, k=min(k, len(X_encoded.columns)))
        else:
            selector = SelectKBest(score_func=f_classif, k=min(k, len(X_encoded.columns)))
        
        X_selected = selector.fit_transform(X_encoded, y)
        selected_features = X_encoded.columns[selector.get_support()].tolist()
        
        self.feature_names[task_type] = selected_features
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features

# ================================
# ML/models/base_model.py
# ================================

from abc import ABC, abstractmethod
import joblib
import pandas as pd
from typing import Any, Dict, Tuple
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np

class BaseModel(ABC):
    """Abstract base class for all ML models"""
    
    def __init__(self, model_name: str, config: ModelConfig):
        self.model_name = model_name
        self.config = config
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

# ================================
# ML/models/regression/advanced_regression.py
# ================================

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
from ..base_model import BaseModel

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
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1
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
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'subsample': [0.8, 0.9, 1.0]
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

# ================================
# ML/models/classification/advanced_classification.py
# ================================

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from ..base_model import BaseModel

class AdvancedClassificationModel(BaseModel):
    """Advanced classification with multiple algorithms"""
    
    def __init__(self, algorithm: str = 'random_forest', config: ModelConfig = None):
        super().__init__(f"{algorithm}_classification", config)
        self.algorithm = algorithm
        self.class_labels = None
    
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
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1
            ),
            'logistic': Pipeline([
                ('scaler', RobustScaler()),
                ('classifier', LogisticRegression(
                    random_state=self.config.RANDOM_STATE,
                    class_weight='balanced'
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
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            },
            'logistic': {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__penalty': ['l1', 'l2']
            },
            'svm': {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__kernel': ['rbf', 'poly']
            }
        }
        
        return grids.get(self.algorithm, {})
    
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

# ================================
# ML/models/time_series/advanced_forecasting.py
# ================================

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesForecaster:
    """Advanced time series forecasting with multiple methods"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
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
                daily_seasonality=False
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

# ================================
# ML/evaluation/metrics.py
# ================================

import pandas as pd
import numpy as np
from sklearn.metrics import *
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """Comprehensive model evaluation and monitoring"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.evaluation_history = []
    
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          model_name: str = "model") -> Dict[str, Any]:
        """Comprehensive regression evaluation"""
        
        # Basic metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        max_error = max_error_score(y_true, y_pred)
        
        # Residual analysis
        residuals = y_true - y_pred
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        
        results = {
            'model_name': model_name,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape,
            'max_error': max_error,
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'evaluation_timestamp': pd.Timestamp.now()
        }
        
        # Generate plots
        self._plot_regression_diagnostics(y_true, y_pred, model_name)
        
        self.evaluation_history.append(results)
        return results
    
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray,
                              y_pred_proba: np.ndarray = None,
                              model_name: str = "model") -> Dict[str, Any]:
        """Comprehensive classification evaluation"""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(y_true, y_pred, zero_division=0),
            'evaluation_timestamp': pd.Timestamp.now()
        }
        
        # ROC AUC for binary classification
        if len(np.unique(y_true)) == 2 and y_pred_proba is not None:
            roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
            results['roc_auc'] = roc_auc
        
        # Generate plots
        self._plot_classification_diagnostics(y_true, y_pred, y_pred_proba, model_name)
        
        self.evaluation_history.append(results)
        return results
    
    def evaluate_time_series(self, y_true: pd.Series, y_pred: pd.Series,
                           model_name: str = "ts_model") -> Dict[str, Any]:
        """Time series specific evaluation"""
        
        # Standard regression metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # Time series specific metrics
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
        
        # Directional accuracy
        direction_true = np.sign(y_true.diff().dropna())
        direction_pred = np.sign(y_pred.diff().dropna())
        directional_accuracy = np.mean(direction_true == direction_pred) * 100
        
        results = {
            'model_name': model_name,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'smape': smape,
            'directional_accuracy': directional_accuracy,
            'evaluation_timestamp': pd.Timestamp.now()
        }
        
        # Generate plots
        self._plot_time_series_diagnostics(y_true, y_pred, model_name)
        
        return results
    
    def _plot_regression_diagnostics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   model_name: str):
        """Generate regression diagnostic plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Predicted vs Actual
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        axes[0, 0].set_xlabel('Actual')
        axes[0, 0].set_ylabel('Predicted')
        axes[0, 0].set_title('Predicted vs Actual')
        
        # Residuals vs Predicted
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        
        # Residual histogram
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residual Distribution')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.PLOTS_DIR, f"{model_name}_regression_diagnostics.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_classification_diagnostics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                       y_pred_proba: np.ndarray, model_name: str):
        """Generate classification diagnostic plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # Class distribution
        unique, counts = np.unique(y_true, return_counts=True)
        axes[0, 1].bar(unique, counts, alpha=0.7)
        axes[0, 1].set_title('Class Distribution')
        axes[0, 1].set_xlabel('Class')
        axes[0, 1].set_ylabel('Count')
        
        # ROC Curve (for binary classification)
        if len(np.unique(y_true)) == 2 and y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            axes[1, 0].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            axes[1, 0].plot([0, 1], [0, 1], 'k--')
            axes[1, 0].set_xlabel('False Positive Rate')
            axes[1, 0].set_ylabel('True Positive Rate')
            axes[1, 0].set_title('ROC Curve')
            axes[1, 0].legend()
        
        # Prediction confidence distribution
        if y_pred_proba is not None:
            max_proba = np.max(y_pred_proba, axis=1)
            axes[1, 1].hist(max_proba, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel('Maximum Prediction Probability')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Prediction Confidence')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.PLOTS_DIR, f"{model_name}_classification_diagnostics.png"),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_time_series_diagnostics(self, y_true: pd.Series, y_pred: pd.Series, 
                                    model_name: str):
        """Generate time series diagnostic plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time series plot
        axes[0, 0].plot(y_true.index, y_true.values, label='Actual', alpha=0.8)
        axes[0, 0].plot(y_pred.index, y_pred.values, label='Predicted', alpha=0.8)
        axes[0, 0].set_title('Time Series: Actual vs Predicted')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Residuals over time
        residuals = y_true - y_pred
        axes[0, 1].plot(y_true.index, residuals, alpha=0.7)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_title('Residuals Over Time')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # ACF of residuals
        try:
            from statsmodels.tsa.stattools import acf
            from statsmodels.graphics.tsaplots import plot_acf
            plot_acf(residuals.dropna(), ax=axes[1, 0], lags=20)
            axes[1, 0].set_title('Residual Autocorrelation')
        except:
            axes[1, 0].hist(residuals.dropna(), bins=20)
            axes[1, 0].set_title('Residual Distribution')
        
        # Error distribution
        errors = np.abs(residuals)
        axes[1, 1].hist(errors, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Absolute Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Error Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.PLOTS_DIR, f"{model_name}_ts_diagnostics.png"),
                   dpi=300, bbox_inches='tight')
        plt.close()

# ================================
# ML/models/ensemble/model_ensemble.py
# ================================

from sklearn.ensemble import VotingRegressor, VotingClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from typing import Dict, List, Any

class ModelEnsemble:
    """Ensemble methods for combining multiple models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.ensemble_model = None
        self.base_models = {}
        self.weights = None
    
    def create_regression_ensemble(self, models: Dict[str, Any]) -> VotingRegressor:
        """Create voting regressor ensemble"""
        estimators = [(name, model) for name, model in models.items()]
        return VotingRegressor(estimators=estimators, n_jobs=-1)
    
    def create_classification_ensemble(self, models: Dict[str, Any]) -> VotingClassifier:
        """Create voting classifier ensemble"""
        estimators = [(name, model) for name, model in models.items()]
        return VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
    
    def train_weighted_ensemble(self, X: pd.DataFrame, y: pd.Series, 
                              models: Dict[str, Any], task_type: str = 'regression') -> Dict:
        """Train ensemble with optimal weights based on CV performance"""
        
        # Get individual model CV scores
        individual_scores = {}
        for name, model in models.items():
            if task_type == 'regression':
                scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
                individual_scores[name] = -scores.mean()  # Convert to positive RMSE
            else:
                scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
                individual_scores[name] = scores.mean()
        
        # Calculate weights (inverse for regression, direct for classification)
        if task_type == 'regression':
            weights = {name: 1/score for name, score in individual_scores.items()}
        else:
            weights = individual_scores.copy()
        
        # Normalize weights
        weight_sum = sum(weights.values())
        self.weights = {name: w/weight_sum for name, w in weights.items()}
        
        # Create and train ensemble
        if task_type == 'regression':
            self.ensemble_model = self.create_regression_ensemble(models)
        else:
            self.ensemble_model = self.create_classification_ensemble(models)
        
        self.ensemble_model.fit(X, y)
        
        return {
            'individual_scores': individual_scores,
            'weights': self.weights,
            'ensemble_trained': True
        }

# ================================
# ML/visualization/interactive_dashboard.py
# ================================

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any

class InteractiveDashboard:
    """Interactive dashboard using Plotly"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.figures = {}
    
    def create_shipment_performance_dashboard(self, shipments: pd.DataFrame) -> go.Figure:
        """Create interactive shipment performance dashboard"""
        
        # Prepare data
        shipments['ShipmentDate'] = pd.to_datetime(shipments['ShipmentDate'], errors='coerce')
        daily_metrics = shipments.groupby(shipments['ShipmentDate'].dt.date).agg({
            'ShipmentID': 'count',
            'OnTimeDeliveryFlag': 'mean',
            'DeliveryVariance': 'mean',
            'ShipmentValue': 'sum'
        }).reset_index()
        daily_metrics.columns = ['Date', 'TotalShipments', 'OnTimeRate', 'AvgVariance', 'TotalValue']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Daily Shipments', 'On-Time Delivery Rate', 
                          'Delivery Variance Trend', 'Shipment Value'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Daily shipments
        fig.add_trace(
            go.Scatter(x=daily_metrics['Date'], y=daily_metrics['TotalShipments'],
                      mode='lines+markers', name='Daily Shipments',
                      line=dict(color='blue')),
            row=1, col=1
        )
        
        # On-time rate
        fig.add_trace(
            go.Scatter(x=daily_metrics['Date'], y=daily_metrics['OnTimeRate']*100,
                      mode='lines+markers', name='On-Time Rate (%)',
                      line=dict(color='green')),
            row=1, col=2
        )
        
        # Delivery variance
        fig.add_trace(
            go.Scatter(x=daily_metrics['Date'], y=daily_metrics['AvgVariance'],
                      mode='lines+markers', name='Avg Delivery Variance (days)',
                      line=dict(color='red')),
            row=2, col=1
        )
        
        # Shipment value
        fig.add_trace(
            go.Scatter(x=daily_metrics['Date'], y=daily_metrics['TotalValue'],
                      mode='lines+markers', name='Total Value',
                      line=dict(color='purple')),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="📦 Shipment Performance Dashboard",
            title_x=0.5,
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_customer_segmentation_dashboard(self, rfm_data: pd.DataFrame) -> go.Figure:
        """Create interactive customer segmentation dashboard"""
        
        # RFM 3D scatter
        fig = go.Figure(data=go.Scatter3d(
            x=rfm_data['Recency'],
            y=rfm_data['Frequency'], 
            z=rfm_data['Monetary'],
            mode='markers',
            marker=dict(
                size=5,
                color=rfm_data.get('segment_kmeans', 0),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Segment")
            ),
            text=rfm_data['CustomerKey'],
            hovertemplate='<b>Customer:</b> %{text}<br>' +
                         '<b>Recency:</b> %{x} days<br>' +
                         '<b>Frequency:</b> %{y} transactions<br>' +
                         '<b>Monetary:</b> $%{z:.2f}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title='👥 Customer RFM Segmentation (3D)',
            scene=dict(
                xaxis_title='Recency (Days)',
                yaxis_title='Frequency (Transactions)',
                zaxis_title='Monetary Value ($)'
            ),
            width=800,
            height=700
        )
        
        return fig
    
    def create_model_performance_comparison(self, evaluation_results: List[Dict]) -> go.Figure:
        """Create model performance comparison dashboard"""
        
        # Prepare data
        model_names = [result['model_name'] for result in evaluation_results]
        
        # Check if we have regression or classification results
        has_r2 = any('r2_score' in result for result in evaluation_results)
        has_accuracy = any('accuracy' in result for result in evaluation_results)
        
        if has_r2:
            # Regression metrics
            r2_scores = [result.get('r2_score', 0) for result in evaluation_results]
            rmse_scores = [result.get('rmse', 0) for result in evaluation_results]
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=['R² Score Comparison', 'RMSE Comparison']
            )
            
            fig.add_trace(
                go.Bar(x=model_names, y=r2_scores, name='R² Score',
                      marker_color='lightblue'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=model_names, y=rmse_scores, name='RMSE',
                      marker_color='lightcoral'),
                row=1, col=2
            )
            
        elif has_accuracy:
            # Classification metrics
            accuracy_scores = [result.get('accuracy', 0) for result in evaluation_results]
            f1_scores = [result.get('f1_score', 0) for result in evaluation_results]
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=['Accuracy Comparison', 'F1 Score Comparison']
            )
            
            fig.add_trace(
                go.Bar(x=model_names, y=accuracy_scores, name='Accuracy',
                      marker_color='lightgreen'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=model_names, y=f1_scores, name='F1 Score',
                      marker_color='lightyellow'),
                row=1, col=2
            )
        
        fig.update_layout(
            title_text="🎯 Model Performance Comparison",
            title_x=0.5,
            height=500
        )
        
        return fig
    
    def create_forecast_dashboard(self, historical: pd.Series, forecast: pd.Series,
                                confidence_interval: Tuple[pd.Series, pd.Series] = None) -> go.Figure:
        """Create interactive forecasting dashboard"""
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical.index,
            y=historical.values,
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode='lines',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        # Confidence interval
        if confidence_interval:
            lower, upper = confidence_interval
            fig.add_trace(go.Scatter(
                x=forecast.index.tolist() + forecast.index.tolist()[::-1],
                y=upper.tolist() + lower.tolist()[::-1],
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                showlegend=True
            ))
        
        fig.update_layout(
            title='📈 Time Series Forecast with Confidence Interval',
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified'
        )
        
        return fig
    
    def save_dashboard_html(self, fig: go.Figure, filename: str):
        """Save interactive dashboard as HTML"""
        filepath = os.path.join(self.config.OUTPUT_DIR, f"{filename}.html")
        fig.write_html(filepath)
        print(f"💾 Dashboard saved: {filepath}")
        return filepath

# ================================
# ML/pipelines/training_pipeline.py
# ================================

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

class MLTrainingPipeline:
    """Comprehensive ML training pipeline"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.results = {}
        self.trained_models = {}
        
    def run_regression_pipeline(self, df: pd.DataFrame, target_col: str,
                              algorithms: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive regression pipeline"""
        
        if algorithms is None:
            algorithms = ['random_forest', 'gradient_boost', 'xgboost', 'ridge']
        
        print(f"🚀 Starting regression pipeline for {target_col}")
        
        # Feature engineering
        feature_engineer = FeatureEngineer()
        df_engineered = feature_engineer.engineer_shipment_features(df)
        
        # Prepare features and target
        y = df_engineered[target_col].dropna()
        X = df_engineered.loc[y.index]
        
        # Feature selection
        X_selected, selected_features = feature_engineer.select_features(
            X.select_dtypes(include=[np.number]), y, task_type='regression'
        )
        
        print(f"📊 Selected {len(selected_features)} features: {selected_features[:5]}...")
        
        # Train multiple models
        model_results = {}
        evaluator = ModelEvaluator(self.config)
        
        for algorithm in algorithms:
            print(f"🤖 Training {algorithm}...")
            
            try:
                # Create and train model
                model = AdvancedRegressionModel(algorithm, self.config)
                
                # Split data
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_selected, y, test_size=self.config.TEST_SIZE, 
                    random_state=self.config.RANDOM_STATE
                )
                
                # Hyperparameter tuning
                tuning_results = model.tune_hyperparameters(X_train, y_train, search_type='random')
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Evaluation
                eval_results = evaluator.evaluate_regression(y_test, y_pred, algorithm)
                
                # Feature importance
                try:
                    feature_importance = model.get_feature_importance()
                except:
                    feature_importance = pd.DataFrame()
                
                model_results[algorithm] = {
                    'evaluation': eval_results,
                    'tuning': tuning_results,
                    'feature_importance': feature_importance,
                    'model_object': model
                }
                
                # Save model
                model_path = model.save_model()
                print(f"💾 Model saved: {model_path}")
                
            except Exception as e:
                print(f"❌ Error training {algorithm}: {e}")
                continue
        
        # Find best model
        best_model = min(model_results.items(), 
                        key=lambda x: x[1]['evaluation']['rmse'])[0]
        
        print(f"🏆 Best model: {best_model}")
        
        self.results[f"{target_col}_regression"] = {
            'best_model': best_model,
            'all_results': model_results,
            'selected_features': selected_features,
            'target_column': target_col
        }
        
        return model_results
    
    def run_classification_pipeline(self, df: pd.DataFrame, target_col: str,
                                  algorithms: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive classification pipeline"""
        
        if algorithms is None:
            algorithms = ['random_forest', 'gradient_boost', 'xgboost', 'logistic']
        
        print(f"🚀 Starting classification pipeline for {target_col}")
        
        # Feature engineering
        feature_engineer = FeatureEngineer()
        if 'ShipmentDate' in df.columns:
            df_engineered = feature_engineer.engineer_shipment_features(df)
        else:
            df_engineered = feature_engineer.engineer_invoice_features(df)
        
        # Prepare features and target
        y = df_engineered[target_col].dropna()
        X = df_engineered.loc[y.index]
        
        # Ensure binary classification
        if target_col == 'OnTimeDeliveryFlag':
            y = (pd.to_numeric(y, errors='coerce') >= 0.5).astype(int)
        elif target_col == 'PaymentStatus':
            y = pd.factorize(y)[0]
        
        # Feature selection
        X_selected, selected_features = feature_engineer.select_features(
            X.select_dtypes(include=[np.number]), y, task_type='classification'
        )
        
        # Train multiple models
        model_results = {}
        evaluator = ModelEvaluator(self.config)
        
        for algorithm in algorithms:
            print(f"🤖 Training {algorithm}...")
            
            try:
                model = AdvancedClassificationModel(algorithm, self.config)
                
                # Split data
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_selected, y, test_size=self.config.TEST_SIZE,
                    random_state=self.config.RANDOM_STATE, stratify=y
                )
                
                # Train model
                cv_results = model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = None
                try:
                    y_pred_proba = model.predict_proba(X_test)
                except:
                    pass
                
                # Evaluation
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
                print(f"❌ Error training {algorithm}: {e}")
                continue
        
        # Find best model
        best_model = max(model_results.items(),
                        key=lambda x: x[1]['evaluation']['accuracy'])[0]
        
        print(f"🏆 Best model: {best_model}")
        
        self.results[f"{target_col}_classification"] = {
            'best_model': best_model,
            'all_results': model_results,
            'selected_features': selected_features,
            'target_column': target_col
        }
        
        return model_results

# ================================
# ML/pipelines/prediction_pipeline.py
# ================================

class PredictionPipeline:
    """Production prediction pipeline"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.loaded_models = {}
    
    def load_production_models(self) -> Dict[str, bool]:
        """Load all production models"""
        model_files = [f for f in os.listdir(self.config.MODEL_DIR) if f.endswith('.joblib')]
        
        results = {}
        for model_file in model_files:
            model_name = model_file.replace('.joblib', '')
            try:
                model_path = os.path.join(self.config.MODEL_DIR, model_file)
                model_data = joblib.load(model_path)
                self.loaded_models[model_name] = model_data
                results[model_name] = True
                print(f"✅ Loaded model: {model_name}")
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                results[model_name] = False
        
        return results
    
    def predict_delivery_performance(self, shipment_data: pd.DataFrame) -> pd.DataFrame:
        """Predict delivery performance for new shipments"""
        
        # Feature engineering
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.engineer_shipment_features(shipment_data)
        
        results = shipment_data.copy()
        
        # On-time delivery prediction
        if 'random_forest_classification' in self.loaded_models:
            model_data = self.loaded_models['random_forest_classification']
            model = model_data['model']
            feature_names = model_data['feature_names']
            
            X = df_features[feature_names].fillna(0)
            
            # Predictions
            ontime_pred = model.predict(X)
            try:
                ontime_proba = model.predict_proba(X)[:, 1]
            except:
                ontime_proba = ontime_pred.astype(float)
            
            results['PredictedOnTimeFlag'] = ontime_pred
            results['OnTimeProbability'] = ontime_proba
        
        # Delivery variance prediction
        if 'random_forest_regression' in self.loaded_models:
            model_data = self.loaded_models['random_forest_regression']
            model = model_data['model']
            feature_names = model_data['feature_names']
            
            X = df_features[feature_names].fillna(0)
            variance_pred = model.predict(X)
            
            results['PredictedDeliveryVariance'] = variance_pred
            results['PredictedArrivalDate'] = (
                pd.to_datetime(results['PlannedArrivalDate']) + 
                pd.to_timedelta(variance_pred, unit='D')
            )
        
        return results
    
    def batch_predict(self, data: pd.DataFrame, model_name: str) -> np.ndarray:
        """Generic batch prediction for any loaded model"""
        if model_name not in self.loaded_models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model_data = self.loaded_models[model_name]
        model = model_data['model']
        feature_names = model_data['feature_names']
        
        # Prepare features
        feature_engineer = FeatureEngineer()
        if 'ShipmentDate' in data.columns:
            data_features = feature_engineer.engineer_shipment_features(data)
        else:
            data_features = feature_engineer.engineer_invoice_features(data)
        
        X = data_features[feature_names].fillna(0)
        predictions = model.predict(X)
        
        return predictions

# ================================
# ML/pipelines/monitoring_pipeline.py
# ================================

import pandas as pd
import numpy as np
from typing import Dict, Any
from scipy import stats

class ModelMonitoringPipeline:
    """Model performance monitoring and drift detection"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.baseline_stats = {}
        self.alerts = []
    
    def detect_data_drift(self, reference_data: pd.DataFrame, 
                         current_data: pd.DataFrame,
                         threshold: float = 0.05) -> Dict[str, Any]:
        """Detect data drift using statistical tests"""
        
        drift_results = {}
        
        # Numeric columns drift detection
        numeric_cols = reference_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in current_data.columns:
                ref_values = reference_data[col].dropna()
                curr_values = current_data[col].dropna()
                
                if len(ref_values) > 0 and len(curr_values) > 0:
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_pvalue = stats.ks_2samp(ref_values, curr_values)
                    
                    # Mann-Whitney U test
                    mw_stat, mw_pvalue = stats.mannwhitneyu(ref_values, curr_values, 
                                                           alternative='two-sided')
                    
                    drift_detected = ks_pvalue < threshold or mw_pvalue < threshold
                    
                    drift_results[col] = {
                        'drift_detected': drift_detected,
                        'ks_statistic': ks_stat,
                        'ks_pvalue': ks_pvalue,
                        'mw_pvalue': mw_pvalue,
                        'ref_mean': ref_values.mean(),
                        'curr_mean': curr_values.mean(),
                        'mean_shift': abs(curr_values.mean() - ref_values.mean()) / ref_values.std()
                    }
                    
                    if drift_detected:
                        self.alerts.append(f"⚠️ Data drift detected in {col}")
        
        return drift_results
    
    def monitor_model_performance(self, model_name: str, y_true: np.ndarray, 
                                y_pred: np.ndarray, task_type: str = 'regression') -> Dict:
        """Monitor model performance over time"""
        
        if task_type == 'regression':
            current_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            current_r2 = r2_score(y_true, y_pred)
            
            # Check against baseline
            if model_name in self.baseline_stats:
                baseline_rmse = self.baseline_stats[model_name]['rmse']
                baseline_r2 = self.baseline_stats[model_name]['r2']
                
                rmse_degradation = (current_rmse - baseline_rmse) / baseline_rmse
                r2_degradation = (baseline_r2 - current_r2) / baseline_r2
                
                if rmse_degradation > 0.1:  # 10% worse
                    self.alerts.append(f"⚠️ {model_name} RMSE degraded by {rmse_degradation:.1%}")
                
                if r2_degradation > 0.1:
                    self.alerts.append(f"⚠️ {model_name} R² degraded by {r2_degradation:.1%}")
            else:
                # Set baseline
                self.baseline_stats[model_name] = {
                    'rmse': current_rmse,
                    'r2': current_r2,
                    'timestamp': datetime.now()
                }
            
            return {
                'current_rmse': current_rmse,
                'current_r2': current_r2,
                'baseline_rmse': self.baseline_stats[model_name]['rmse'],
                'baseline_r2': self.baseline_stats[model_name]['r2']
            }
        
        else:  # classification
            current_accuracy = accuracy_score(y_true, y_pred)
            current_f1 = f1_score(y_true, y_pred, average='weighted')
            
            if model_name in self.baseline_stats:
                baseline_acc = self.baseline_stats[model_name]['accuracy']
                acc_degradation = (baseline_acc - current_accuracy) / baseline_acc
                
                if acc_degradation > 0.05:  # 5% worse
                    self.alerts.append(f"⚠️ {model_name} accuracy degraded by {acc_degradation:.1%}")
            else:
                self.baseline_stats[model_name] = {
                    'accuracy': current_accuracy,
                    'f1_score': current_f1,
                    'timestamp': datetime.now()
                }
            
            return {
                'current_accuracy': current_accuracy,
                'current_f1': current_f1,
                'baseline_accuracy': self.baseline_stats[model_name]['accuracy']
            }

# ================================
# ML/enhanced_customer_segmentation.py
# ================================

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from typing import Dict, Tuple

class AdvancedCustomerSegmentation:
    """Enhanced customer segmentation with multiple algorithms"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.cluster_models = {}
    
    def compute_enhanced_rfm(self, invoices_df: pd.DataFrame, 
                           snapshot_date: pd.Timestamp = None) -> pd.DataFrame:
        """Enhanced RFM calculation with additional business metrics"""
        
        if snapshot_date is None:
            snapshot_date = pd.Timestamp.now()
        
        df = invoices_df.copy()
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        
        # Basic RFM
        rfm = df.groupby('CustomerKey').agg({
            'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
            'InvoiceID': 'nunique',
            'NetAmount': ['sum', 'mean', 'std']
        })
        
        # Flatten columns
        rfm.columns = ['Recency', 'Frequency', 'Monetary_sum', 'Monetary_mean', 'Monetary_std']
        rfm['Monetary'] = rfm['Monetary_sum']
        rfm['Monetary_std'] = rfm['Monetary_std'].fillna(0)
        
        # Additional metrics
        customer_details = df.groupby('CustomerKey').agg({
            'InvoiceDate': ['min', 'max'],
            'PaymentStatus': lambda x: (x == 1).mean() if len(x) > 0 else 0,  # Payment rate
            'TotalAmount': 'sum'
        })
        
        customer_details.columns = ['FirstPurchase', 'LastPurchase', 'PaymentRate', 'TotalSpent']
        
        # Customer lifetime metrics
        customer_details['CustomerLifetime'] = (customer_details['LastPurchase'] - 
                                              customer_details['FirstPurchase']).dt.days
        customer_details['AvgOrderValue'] = customer_details['TotalSpent'] / rfm['Frequency']
        customer_details['PurchaseVelocity'] = rfm['Frequency'] / (customer_details['CustomerLifetime'] + 1)
        
        # Combine all metrics
        enhanced_rfm = rfm.join(customer_details)
        enhanced_rfm = enhanced_rfm.fillna(0)
        
        # RFM scoring with enhanced logic
        enhanced_rfm['R_score'] = pd.qcut(enhanced_rfm['Recency'].rank(method='first'), 
                                         q=5, labels=[5,4,3,2,1]).astype(int)
        enhanced_rfm['F_score'] = pd.qcut(enhanced_rfm['Frequency'].rank(method='first'), 
                                         q=5, labels=[1,2,3,4,5]).astype(int)
        enhanced_rfm['M_score'] = pd.qcut(enhanced_rfm['Monetary'].rank(method='first'), 
                                         q=5, labels=[1,2,3,4,5]).astype(int)
        
        # Customer value score
        enhanced_rfm['CustomerValue'] = (enhanced_rfm['F_score'] * enhanced_rfm['M_score'] * 
                                       enhanced_rfm['PaymentRate'])
        
        return enhanced_rfm.reset_index()
    
    def optimal_clustering(self, data: pd.DataFrame, 
                         features: List[str], max_clusters: int = 10) -> Dict:
        """Find optimal number of clusters using multiple metrics"""
        
        X = data[features].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        cluster_metrics = {}
        
        for k in range(2, max_clusters + 1):
            # KMeans
            kmeans = KMeans(n_clusters=k, random_state=self.config.RANDOM_STATE, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            # Calculate metrics
            silhouette = silhouette_score(X_scaled, labels)
            calinski = calinski_harabasz_score(X_scaled, labels)
            inertia = kmeans.inertia_
            
            cluster_metrics[k] = {
                'silhouette_score': silhouette,
                'calinski_harabasz': calinski,
                'inertia': inertia,
                'model': kmeans
            }
        
        # Find optimal k using silhouette score
        optimal_k = max(cluster_metrics.keys(), 
                       key=lambda k: cluster_metrics[k]['silhouette_score'])
        
        print(f"🎯 Optimal number of clusters: {optimal_k}")
        print(f"   Silhouette Score: {cluster_metrics[optimal_k]['silhouette_score']:.3f}")
        
        return cluster_metrics, optimal_k
    
    def advanced_segmentation(self, rfm_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Advanced customer segmentation with multiple algorithms"""
        
        # Features for clustering
        clustering_features = ['Recency', 'Frequency', 'Monetary', 'CustomerValue', 
                             'AvgOrderValue', 'PurchaseVelocity', 'PaymentRate']
        
        available_features = [f for f in clustering_features if f in rfm_data.columns]
        
        # Find optimal clusters
        cluster_metrics, optimal_k = self.optimal_clustering(rfm_data, available_features)
        
        # Apply optimal clustering
        X = rfm_data[available_features].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # KMeans with optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=self.config.RANDOM_STATE, n_init=10)
        rfm_data['Segment_KMeans'] = kmeans.fit_predict(X_scaled)
        
        # DBSCAN for outlier detection
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        rfm_data['Segment_DBSCAN'] = dbscan.fit_predict(X_scaled)
        
        # Business rule-based segmentation
        rfm_data['Segment_Business'] = self._business_rule_segmentation(rfm_data)
        
        # Segment profiling
        segment_profiles = self._profile_segments(rfm_data, 'Segment_KMeans')
        
        results = {
            'optimal_clusters': optimal_k,
            'cluster_metrics': cluster_metrics,
            'segment_profiles': segment_profiles,
            'features_used': available_features
        }
        
        return rfm_data, results
    
    def _business_rule_segmentation(self, rfm_data: pd.DataFrame) -> pd.Series:
        """Business rule-based customer segmentation"""
        
        conditions = [
            (rfm_data['R_score'] >= 4) & (rfm_data['F_score'] >= 4) & (rfm_data['M_score'] >= 4),
            (rfm_data['R_score'] >= 3) & (rfm_data['F_score'] >= 3) & (rfm_data['M_score'] >= 3),
            (rfm_data['R_score'] >= 3) & (rfm_data['F_score'] <= 2),
            (rfm_data['R_score'] <= 2) & (rfm_data['F_score'] >= 3) & (rfm_data['M_score'] >= 3),
            (rfm_data['R_score'] <= 2) & (rfm_data['F_score'] <= 2)
        ]
        
        choices = ['Champions', 'Loyal_Customers', 'New_Customers', 'At_Risk', 'Lost_Customers']
        
        return pd.Series(np.select(conditions, choices, default='Others'), 
                        index=rfm_data.index)
    
    def _profile_segments(self, data: pd.DataFrame, segment_col: str) -> Dict:
        """Create detailed segment profiles"""
        
        profiles = {}
        
        for segment in data[segment_col].unique():
            segment_data = data[data[segment_col] == segment]
            
            profiles[segment] = {
                'size': len(segment_data),
                'percentage': len(segment_data) / len(data) * 100,
                'avg_recency': segment_data['Recency'].mean(),
                'avg_frequency': segment_data['Frequency'].mean(),
                'avg_monetary': segment_data['Monetary'].mean(),
                'avg_payment_rate': segment_data.get('PaymentRate', pd.Series([0])).mean(),
                'total_value': segment_data['Monetary'].sum(),
                'characteristics': self._get_segment_characteristics(segment_data)
            }
        
        return profiles
    
    def _get_segment_characteristics(self, segment_data: pd.DataFrame) -> Dict:
        """Get detailed characteristics of a segment"""
        
        return {
            'high_value_customers': (segment_data['Monetary'] > segment_data['Monetary'].quantile(0.8)).sum(),
            'frequent_buyers': (segment_data['Frequency'] > segment_data['Frequency'].median()).sum(),
            'recent_activity': (segment_data['Recency'] < 30).sum(),
            'payment_reliability': segment_data.get('PaymentRate', pd.Series([0])).mean()
        }

# ================================
# ML/advanced_time_series.py
# ================================

class AdvancedTimeSeriesAnalysis:
    """Advanced time series analysis and forecasting"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.forecaster = TimeSeriesForecaster(config)
    
    def comprehensive_forecast(self, df: pd.DataFrame, date_col: str, 
                             value_col: str, horizon: int = 30) -> Dict[str, Any]:
        """Comprehensive time series forecasting with multiple methods"""
        
        # Prepare time series
        ts = self.forecaster.prepare_time_series(df, date_col, value_col)
        
        if len(ts) < 14:
            print("⚠️ Insufficient data for time series analysis")
            return None
        
        print(f"📊 Time series length: {len(ts)} periods")
        print(f"📅 Date range: {ts.index.min().date()} to {ts.index.max().date()}")
        
        # Generate forecasts using different methods
        forecasting_results = {}
        
        # Exponential Smoothing
        exp_forecast, exp_metrics = self.forecaster.exponential_smoothing_forecast(ts, horizon)
        forecasting_results['exponential_smoothing'] = {
            'forecast': exp_forecast,
            'metrics': exp_metrics
        }
        
        # ARIMA
        arima_forecast, arima_metrics = self.forecaster.arima_forecast(ts, horizon)
        forecasting_results['arima'] = {
            'forecast': arima_forecast,
            'metrics': arima_metrics
        }
        
        # Ensemble
        ensemble_forecast, ensemble_metrics = self.forecaster.ensemble_forecast(ts, horizon)
        forecasting_results['ensemble'] = {
            'forecast': ensemble_forecast,
            'metrics': ensemble_metrics
        }
        
        # Seasonality analysis
        seasonality_analysis = self._analyze_seasonality(ts)
        
        # Trend analysis
        trend_analysis = self._analyze_trend(ts)
        
        # Save results
        forecast_df = pd.DataFrame({
            'date': list(ts.index) + list(ensemble_forecast.index),
            'actual': list(ts.values) + [np.nan] * len(ensemble_forecast),
            'forecast': [np.nan] * len(ts) + list(ensemble_forecast.values),
            'type': ['historical'] * len(ts) + ['forecast'] * len(ensemble_forecast)
        })
        
        output_path = os.path.join(self.config.OUTPUT_DIR, f"forecast_{value_col}.csv")
        forecast_df.to_csv(output_path, index=False)
        
        return {
            'historical_data': ts,
            'forecasts': forecasting_results,
            'seasonality': seasonality_analysis,
            'trend': trend_analysis,
            'best_method': self._select_best_method(forecasting_results, ts),
            'output_file': output_path
        }
    
    def _analyze_seasonality(self, ts: pd.Series) -> Dict:
        """Analyze seasonality patterns"""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            if len(ts) >= 24:  # Need enough data for decomposition
                decomposition = seasonal_decompose(ts, model='additive', period=7)
                
                return {
                    'has_seasonality': True,
                    'seasonal_strength': np.var(decomposition.seasonal) / np.var(ts),
                    'trend_strength': np.var(decomposition.trend.dropna()) / np.var(ts),
                    'residual_variance': np.var(decomposition.resid.dropna())
                }
            else:
                return {'has_seasonality': False, 'reason': 'insufficient_data'}
                
        except Exception as e:
            return {'has_seasonality': False, 'error': str(e)}
    
    def _analyze_trend(self, ts: pd.Series) -> Dict:
        """Analyze trend patterns"""
        try:
            # Linear trend
            x = np.arange(len(ts))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, ts.values)
            
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
    
    def _select_best_method(self, forecasting_results: Dict, historical_ts: pd.Series) -> str:
        """Select best forecasting method based on validation"""
        
        if len(historical_ts) < 21:  # Not enough for validation
            return 'ensemble'
        
        # Use last 7 days as validation
        train_ts = historical_ts[:-7]
        val_ts = historical_ts[-7:]
        
        method_errors = {}
        
        for method_name, result in forecasting_results.items():
            try:
                if method_name == 'exponential_smoothing':
                    val_forecast, _ = self.forecaster.exponential_smoothing_forecast(train_ts, 7)
                elif method_name == 'arima':
                    val_forecast, _ = self.forecaster.arima_forecast(train_ts, 7)
                else:
                    continue
                
                # Calculate MAPE on validation set
                mape = np.mean(np.abs((val_ts.values - val_forecast.values) / 
                                    (val_ts.values + 1e-8))) * 100
                method_errors[method_name] = mape
                
            except Exception as e:
                print(f"⚠️ Validation failed for {method_name}: {e}")
                continue
        
        if method_errors:
            best_method = min(method_errors.items(), key=lambda x: x[1])[0]
            print(f"🎯 Best method: {best_method} (MAPE: {method_errors[best_method]:.2f}%)")
            return best_method
        else:
            return 'ensemble'

# ================================
# ML/enhanced_main_pipeline.py
# ================================

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, Any, Optional

class EnhancedMLPipeline:
    """Enhanced ML pipeline with comprehensive analytics"""
    
    def __init__(self, sample_limit: Optional[int] = None):
        self.config = ModelConfig()
        self.sample_limit = sample_limit
        self.results = {}
        self.data_loader = DataLoader()
        self.dashboard = InteractiveDashboard(self.config)
        
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete enhanced ML pipeline"""
        
        print("🚀 Starting Enhanced ML Pipeline for TRALIS Analytics")
        print("=" * 60)
        
        # 1. Data Loading
        print("\n📥 STEP 1: Enhanced Data Loading")
        shipments = self.data_loader.load_shipments(self.sample_limit)
        invoices = self.data_loader.load_invoices(self.sample_limit)
        
        if shipments.empty or invoices.empty:
            raise ValueError("❌ No data loaded - check database connection")
        
        data_summary = self.data_loader.get_data_summary()
        print(f"✅ Data loaded successfully: {data_summary}")
        
        # 2. Advanced Preprocessing & Feature Engineering
        print("\n🔧 STEP 2: Advanced Feature Engineering")
        feature_engineer = FeatureEngineer()
        
        shipments_enhanced = feature_engineer.engineer_shipment_features(shipments)
        invoices_enhanced = feature_engineer.engineer_invoice_features(invoices)
        
        print(f"🔹 Shipment features: {len(shipments_enhanced.columns)} (added {len(shipments_enhanced.columns) - len(shipments.columns)} new)")
        print(f"🔹 Invoice features: {len(invoices_enhanced.columns)} (added {len(invoices_enhanced.columns) - len(invoices.columns)} new)")
        
        # 3. Regression Models
        print("\n🤖 STEP 3: Advanced Regression Models")
        training_pipeline = MLTrainingPipeline(self.config)
        
        # Delivery variance prediction
        if 'DeliveryVariance' in shipments_enhanced.columns:
            delivery_results = training_pipeline.run_regression_pipeline(
                shipments_enhanced, 'DeliveryVariance',
                algorithms=['random_forest', 'xgboost', 'gradient_boost', 'ridge']
            )
            self.results['delivery_variance_models'] = delivery_results
        
        # Invoice profit prediction
        if 'InvoiceProfit' in invoices_enhanced.columns or 'NetAmount' in invoices_enhanced.columns:
            # Create profit target if not exists
            if 'InvoiceProfit' not in invoices_enhanced.columns:
                invoices_enhanced['InvoiceProfit'] = invoices_enhanced['NetAmount'] * 0.3  # 30% margin estimate
            
            profit_results = training_pipeline.run_regression_pipeline(
                invoices_enhanced, 'InvoiceProfit',
                algorithms=['random_forest', 'xgboost', 'gradient_boost', 'lasso']
            )
            self.results['invoice_profit_models'] = profit_results
        
        # 4. Classification Models
        print("\n🧠 STEP 4: Advanced Classification Models")
        
        # On-time delivery prediction
        if 'OnTimeDeliveryFlag' in shipments_enhanced.columns:
            ontime_results = training_pipeline.run_classification_pipeline(
                shipments_enhanced, 'OnTimeDeliveryFlag',
                algorithms=['random_forest', 'xgboost', 'gradient_boost', 'logistic']
            )
            self.results['ontime_delivery_models'] = ontime_results
        
        # Payment status prediction
        if 'PaymentStatus' in invoices_enhanced.columns:
            payment_results = training_pipeline.run_classification_pipeline(
                invoices_enhanced, 'PaymentStatus',
                algorithms=['random_forest', 'xgboost', 'logistic']
            )
            self.results['payment_status_models'] = payment_results
        
        # 5. Advanced Time Series Forecasting
        print("\n📈 STEP 5: Advanced Time Series Forecasting")
        ts_analyzer = AdvancedTimeSeriesAnalysis(self.config)
        
        # Daily shipment forecasting
        shipment_ts_results = ts_analyzer.comprehensive_forecast(
            shipments_enhanced, 'ShipmentDate', 'ShipmentValue', horizon=self.config.TS_HORIZON_DAYS
        )
        self.results['shipment_forecasting'] = shipment_ts_results
        
        # Monthly invoice forecasting
        invoice_ts_results = ts_analyzer.comprehensive_forecast(
            invoices_enhanced, 'InvoiceDate', 'NetAmount', horizon=30  # 30 days
        )
        self.results['invoice_forecasting'] = invoice_ts_results
        
        # 6. Advanced Customer Segmentation
        print("\n👥 STEP 6: Advanced Customer Segmentation")
        segmentation = AdvancedCustomerSegmentation(self.config)
        
        enhanced_rfm = segmentation.compute_enhanced_rfm(invoices_enhanced)
        segmented_customers, segmentation_results = segmentation.advanced_segmentation(enhanced_rfm)
        
        self.results['customer_segmentation'] = {
            'rfm_data': enhanced_rfm,
            'segmented_data': segmented_customers,
            'analysis_results': segmentation_results
        }
        
        # 7. Profitability Analysis
        print("\n💰 STEP 7: Enhanced Profitability Analysis")
        profitability_results = self._advanced_profitability_analysis(
            invoices_enhanced, shipments_enhanced, segmented_customers
        )
        self.results['profitability_analysis'] = profitability_results
        
        # 8. Interactive Dashboards
        print("\n📊 STEP 8: Interactive Dashboard Generation")
        self._generate_interactive_dashboards(
            shipments_enhanced, invoices_enhanced, segmented_customers
        )
        
        # 9. Model Monitoring Setup
        print("\n🔍 STEP 9: Model Monitoring Setup")
        monitoring_results = self._setup_model_monitoring(shipments_enhanced, invoices_enhanced)
        self.results['monitoring'] = monitoring_results
        
        # 10. Generate Comprehensive Report
        print("\n📋 STEP 10: Generating Comprehensive Report")
        final_report = self._generate_final_report()
        
        print("\n🎉 Enhanced ML Pipeline Completed Successfully!")
        print("=" * 60)
        print(f"📊 Total models trained: {self._count_trained_models()}")
        print(f"📈 Dashboards created: {len(os.listdir(self.config.OUTPUT_DIR))}")
        print(f"💾 Results saved to: {self.config.OUTPUT_DIR}")
        
        return final_report
    
    def _advanced_profitability_analysis(self, invoices: pd.DataFrame, 
                                       shipments: pd.DataFrame,
                                       segmented_customers: pd.DataFrame) -> Dict:
        """Enhanced profitability analysis with segmentation insights"""
        
        # Customer profitability with segments
        customer_profit = invoices.groupby('CustomerKey').agg({
            'NetAmount': ['sum', 'mean', 'count'],
            'TaxAmount': 'sum',
            'InvoiceProfit': 'sum' if 'InvoiceProfit' in invoices.columns else 'mean'
        }).round(2)
        
        customer_profit.columns = ['TotalRevenue', 'AvgOrderValue', 'OrderCount', 
                                 'TotalTax', 'TotalProfit']
        customer_profit['ProfitMargin'] = (customer_profit['TotalProfit'] / 
                                         customer_profit['TotalRevenue']).fillna(0)
        
        # Merge with segments
        profit_with_segments = customer_profit.merge(
            segmented_customers[['CustomerKey', 'Segment_KMeans', 'Segment_Business']],
            on='CustomerKey', how='left'
        )
        
        # Segment profitability analysis
        segment_profit = profit_with_segments.groupby('Segment_Business').agg({
            'TotalRevenue': ['sum', 'mean'],
            'TotalProfit': ['sum', 'mean'],
            'OrderCount': 'sum',
            'ProfitMargin': 'mean'
        }).round(2)
        
        # Customer lifetime value estimation
        customer_ltv = self._estimate_customer_ltv(invoices, segmented_customers)
        
        return {
            'customer_profitability': customer_profit,
            'segment_profitability': segment_profit,
            'customer_ltv': customer_ltv,
            'top_customers': customer_profit.nlargest(20, 'TotalRevenue'),
            'most_profitable_segments': segment_profit.nlargest(5, ('TotalProfit', 'sum'))
        }
    
    def _estimate_customer_ltv(self, invoices: pd.DataFrame, 
                             segmented_customers: pd.DataFrame) -> pd.DataFrame:
        """Estimate Customer Lifetime Value"""
        
        # Customer metrics for LTV calculation
        customer_metrics = invoices.groupby('CustomerKey').agg({
            'NetAmount': ['sum', 'mean'],
            'InvoiceDate': ['min', 'max', 'count']
        })
        
        customer_metrics.columns = ['TotalSpent', 'AvgOrderValue', 'FirstOrder', 'LastOrder', 'OrderCount']
        
        # Calculate customer age and purchase frequency
        customer_metrics['CustomerAge'] = (customer_metrics['LastOrder'] - 
                                         customer_metrics['FirstOrder']).dt.days
        customer_metrics['PurchaseFrequency'] = (customer_metrics['OrderCount'] / 
                                               (customer_metrics['CustomerAge'] + 1))
        
        # Simple LTV calculation: AOV * Purchase Frequency * Predicted Lifespan
        customer_metrics['PredictedLifespan'] = customer_metrics['CustomerAge'] * 1.5  # Assume 50% growth
        customer_metrics['EstimatedLTV'] = (customer_metrics['AvgOrderValue'] * 
                                          customer_metrics['PurchaseFrequency'] * 
                                          customer_metrics['PredictedLifespan'])
        
        # Merge with segments for LTV by segment analysis
        ltv_with_segments = customer_metrics.merge(
            segmented_customers[['CustomerKey', 'Segment_Business']], 
            on='CustomerKey', how='left'
        )
        
        return ltv_with_segments
    
    def _generate_interactive_dashboards(self, shipments: pd.DataFrame, 
                                       invoices: pd.DataFrame,
                                       segmented_customers: pd.DataFrame):
        """Generate all interactive dashboards"""
        
        # Shipment performance dashboard
        shipment_fig = self.dashboard.create_shipment_performance_dashboard(shipments)
        self.dashboard.save_dashboard_html(shipment_fig, "shipment_performance_dashboard")
        
        # Customer segmentation dashboard
        rfm_fig = self.dashboard.create_customer_segmentation_dashboard(segmented_customers)
        self.dashboard.save_dashboard_html(rfm_fig, "customer_segmentation_dashboard")
        
        # Model performance comparison
        if hasattr(self, 'evaluation_results'):
            model_fig = self.dashboard.create_model_performance_comparison(self.evaluation_results)
            self.dashboard.save_dashboard_html(model_fig, "model_performance_dashboard")
        
        # Forecasting dashboard
        if 'shipment_forecasting' in self.results:
            forecast_data = self.results['shipment_forecasting']
            if forecast_data and 'forecasts' in forecast_data:
                ensemble_forecast = forecast_data['forecasts']['ensemble']['forecast']
                historical = forecast_data['historical_data']
                
                forecast_fig = self.dashboard.create_forecast_dashboard(historical, ensemble_forecast)
                self.dashboard.save_dashboard_html(forecast_fig, "forecasting_dashboard")
        
        print("✅ Interactive dashboards generated and saved as HTML files")
    
    def _setup_model_monitoring(self, shipments: pd.DataFrame, 
                              invoices: pd.DataFrame) -> Dict:
        """Setup model monitoring and drift detection"""
        
        monitoring = ModelMonitoringPipeline(self.config)
        
        # Set baseline statistics for monitoring
        monitoring.baseline_stats = {
            'shipment_data': {
                'mean_weight': shipments['TotalWeight'].mean(),
                'mean_value': shipments['ShipmentValue'].mean(),
                'ontime_rate': shipments.get('OnTimeDeliveryFlag', pd.Series([0.8])).mean()
            },
            'invoice_data': {
                'mean_amount': invoices['NetAmount'].mean(),
                'payment_rate': invoices.get('PaymentStatus', pd.Series([1])).mean()
            }
        }
        
        # Save monitoring configuration
        monitoring_config = {
            'baseline_timestamp': datetime.now().isoformat(),
            'baseline_stats': monitoring.baseline_stats,
            'monitoring_thresholds': {
                'drift_threshold': 0.05,
                'performance_degradation_threshold': 0.1
            }
        }
        
        config_path = os.path.join(self.config.OUTPUT_DIR, "monitoring_config.json")
        with open(config_path, 'w') as f:
            json.dump(monitoring_config, f, indent=2, default=str)
        
        return monitoring_config
    
    def _count_trained_models(self) -> int:
        """Count total number of trained models"""
        count = 0
        for key, value in self.results.items():
            if isinstance(value, dict) and 'all_results' in value:
                count += len(value['all_results'])
        return count
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        
        report = {
            'pipeline_info': {
                'execution_timestamp': datetime.now().isoformat(),
                'pipeline_version': '2.0_enhanced',
                'data_sample_limit': self.sample_limit,
                'total_models_trained': self._count_trained_models()
            },
            'data_summary': self.data_loader.get_data_summary(),
            'model_results': {},
            'business_insights': {},
            'recommendations': []
        }
        
        # Extract key model performance metrics
        for pipeline_name, pipeline_results in self.results.items():
            if isinstance(pipeline_results, dict) and 'best_model' in pipeline_results:
                best_model = pipeline_results['best_model']
                best_results = pipeline_results['all_results'][best_model]['evaluation']
                
                report['model_results'][pipeline_name] = {
                    'best_algorithm': best_model,
                    'performance_metrics': best_results,
                    'feature_count': len(pipeline_results.get('selected_features', []))
                }
        
        # Business insights
        if 'customer_segmentation' in self.results:
            seg_results = self.results['customer_segmentation']['analysis_results']
            report['business_insights']['customer_segments'] = {
                'total_segments': seg_results['optimal_clusters'],
                'segment_profiles': seg_results['segment_profiles']
            }
        
        if 'profitability_analysis' in self.results:
            profit_results = self.results['profitability_analysis']
            report['business_insights']['profitability'] = {
                'top_revenue_customer': profit_results['customer_profitability'].idxmax()['TotalRevenue'],
                'avg_profit_margin': profit_results['customer_profitability']['ProfitMargin'].mean()
            }
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations()
        
        # Save comprehensive report
        report_path = os.path.join(self.config.REPORTS_DIR, "comprehensive_ml_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate executive summary
        self._generate_executive_summary(report)
        
        print(f"📊 Comprehensive report saved: {report_path}")
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate business recommendations based on analysis results"""
        
        recommendations = []
        
        # Model performance recommendations
        for pipeline_name, pipeline_results in self.results.items():
            if isinstance(pipeline_results, dict) and 'best_model' in pipeline_results:
                best_model = pipeline_results['best_model']
                best_score = pipeline_results['all_results'][best_model]['evaluation']
                
                if pipeline_name.endswith('_regression'):
                    if best_score.get('r2_score', 0) < self.config.MIN_R2:
                        recommendations.append(
                            f"⚠️ {pipeline_name}: R² score ({best_score.get('r2_score', 0):.3f}) "
                            f"below threshold ({self.config.MIN_R2}). Consider more data or feature engineering."
                        )
                    else:
                        recommendations.append(
                            f"✅ {pipeline_name}: Good performance (R² = {best_score.get('r2_score', 0):.3f}). "
                            f"Best algorithm: {best_model}"
                        )
                
                elif pipeline_name.endswith('_classification'):
                    if best_score.get('accuracy', 0) < self.config.MIN_ACCURACY:
                        recommendations.append(
                            f"⚠️ {pipeline_name}: Accuracy ({best_score.get('accuracy', 0):.3f}) "
                            f"below threshold ({self.config.MIN_ACCURACY}). Consider class balancing."
                        )
                    else:
                        recommendations.append(
                            f"✅ {pipeline_name}: Good performance (Accuracy = {best_score.get('accuracy', 0):.3f}). "
                            f"Best algorithm: {best_model}"
                        )
        
        # Business recommendations
        if 'customer_segmentation' in self.results:
            seg_results = self.results['customer_segmentation']['analysis_results']
            recommendations.append(
                f"👥 Customer base segmented into {seg_results['optimal_clusters']} distinct groups. "
                "Focus marketing efforts on high-value segments."
            )
        
        # Time series recommendations
        if 'shipment_forecasting' in self.results:
            ts_results = self.results['shipment_forecasting']
            if ts_results:
                best_method = ts_results.get('best_method', 'unknown')
                recommendations.append(
                    f"📈 Time series forecasting: {best_method} method performs best. "
                    "Use for demand planning and resource allocation."
                )
        
        return recommendations
    
    def _generate_executive_summary(self, report: Dict[str, Any]):
        """Generate executive summary in markdown format"""
        
        summary_md = f"""# 📊 TRALIS ML Analytics - Executive Summary

## 📈 Pipeline Execution Summary
- **Execution Date**: {report['pipeline_info']['execution_timestamp'][:10]}
- **Models Trained**: {report['pipeline_info']['total_models_trained']}
- **Data Records Processed**: {sum(summary['rows'] for summary in report['data_summary'].values())}

## 🎯 Key Performance Metrics

### Model Performance
"""
        
        for model_name, model_info in report['model_results'].items():
            algorithm = model_info['best_algorithm']
            metrics = model_info['performance_metrics']
            
            if 'r2_score' in metrics:
                summary_md += f"- **{model_name}**: {algorithm} (R² = {metrics['r2_score']:.3f}, RMSE = {metrics['rmse']:.2f})\n"
            elif 'accuracy' in metrics:
                summary_md += f"- **{model_name}**: {algorithm} (Accuracy = {metrics['accuracy']:.3f})\n"
        
        summary_md += "\n### Business Insights\n"
        
        if 'customer_segments' in report['business_insights']:
            seg_count = report['business_insights']['customer_segments']['total_segments']
            summary_md += f"- **Customer Segmentation**: {seg_count} distinct customer segments identified\n"
        
        if 'profitability' in report['business_insights']:
            avg_margin = report['business_insights']['profitability']['avg_profit_margin']
            summary_md += f"- **Average Profit Margin**: {avg_margin:.1%}\n"
        
        summary_md += "\n## 🚀 Key Recommendations\n\n"
        for i, rec in enumerate(report['recommendations'], 1):
            summary_md += f"{i}. {rec}\n"
        
        summary_md += f"""
## 📁 Output Files Generated
- Interactive dashboards (HTML): `{self.config.OUTPUT_DIR}/*.html`
- Model files: `{self.config.MODEL_DIR}/*.joblib`
- Diagnostic plots: `{self.config.PLOTS_DIR}/*.png`
- Data outputs: `{self.config.OUTPUT_DIR}/*.csv`

## 🔄 Next Steps
1. Review model performance metrics and validate business relevance
2. Deploy best-performing models to production environment
3. Set up automated monitoring and retraining schedules
4. Integrate insights into business decision-making processes
"""
        
        # Save executive summary
        summary_path = os.path.join(self.config.REPORTS_DIR, "executive_summary.md")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_md)
        
        print(f"📋 Executive summary saved: {summary_path}")

# ================================
# ML/run_enhanced_pipeline.py
# ================================

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_main_pipeline import EnhancedMLPipeline
import argparse

def main():
    """Main entry point for enhanced ML pipeline"""
    
    parser = argparse.ArgumentParser(description="Enhanced TRALIS ML Pipeline")
    parser.add_argument("--sample", type=int, default=None, 
                       help="Limit number of records for testing (default: all data)")
    parser.add_argument("--models", nargs='+', 
                       choices=['regression', 'classification', 'timeseries', 'segmentation', 'all'],
                       default=['all'], help="Which model types to run")
    parser.add_argument("--tune", action='store_true', 
                       help="Enable hyperparameter tuning (slower)")
    parser.add_argument("--dashboards", action='store_true', default=True,
                       help="Generate interactive dashboards")
    
    args = parser.parse_args()
    
    try:
        # Initialize and run pipeline
        pipeline = EnhancedMLPipeline(sample_limit=args.sample)
        
        if 'all' in args.models:
            results = pipeline.run_complete_pipeline()
        else:
            # Run specific components
            results = pipeline.run_selective_pipeline(args.models)
        
        print("\n🎉 Pipeline execution completed successfully!")
        print(f"📊 Check results in: {pipeline.config.OUTPUT_DIR}")
        
        return results
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()

# ================================
# ML/utils/model_utils.py
# ================================

import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, List
import json
from datetime import datetime

class ModelUtils:
    """Utility functions for model management"""
    
    @staticmethod
    def compare_models(model_results: Dict[str, Dict]) -> pd.DataFrame:
        """Compare multiple model results in a structured format"""
        
        comparison_data = []
        
        for model_name, results in model_results.items():
            eval_metrics = results.get('evaluation', {})
            
            comparison_data.append({
                'Model': model_name,
                'Algorithm': results.get('algorithm', 'unknown'),
                'R²_Score': eval_metrics.get('r2_score', np.nan),
                'RMSE': eval_metrics.get('rmse', np.nan),
                'MAE': eval_metrics.get('mae', np.nan),
                'Accuracy': eval_metrics.get('accuracy', np.nan),
                'F1_Score': eval_metrics.get('f1_score', np.nan),
                'Training_Time': eval_metrics.get('training_time', np.nan)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Rank models
        if not comparison_df['R²_Score'].isna().all():
            comparison_df['R²_Rank'] = comparison_df['R²_Score'].rank(ascending=False)
        if not comparison_df['Accuracy'].isna().all():
            comparison_df['Accuracy_Rank'] = comparison_df['Accuracy'].rank(ascending=False)
        
        return comparison_df.sort_values(by='R²_Score', ascending=False, na_position='last')
    
    @staticmethod
    def export_model_metadata(model_results: Dict, output_path: str):
        """Export model metadata for documentation"""
        
        metadata = {
            'export_timestamp': datetime.now().isoformat(),
            'models': {}
        }
        
        for model_name, results in model_results.items():
            metadata['models'][model_name] = {
                'algorithm': results.get('algorithm', 'unknown'),
                'performance': results.get('evaluation', {}),
                'features_used': results.get('selected_features', []),
                'hyperparameters': results.get('tuning', {}).get('best_params', {}),
                'cross_validation': results.get('cv_results', {})
            }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"📋 Model metadata exported: {output_path}")
    
    @staticmethod
    def generate_feature_importance_report(model_results: Dict) -> pd.DataFrame:
        """Generate consolidated feature importance report"""
        
        all_importances = []
        
        for model_name, results in model_results.items():
            if 'feature_importance' in results and not results['feature_importance'].empty:
                importance_df = results['feature_importance'].copy()
                importance_df['model'] = model_name
                importance_df['rank'] = range(1, len(importance_df) + 1)
                all_importances.append(importance_df)
        
        if all_importances:
            consolidated = pd.concat(all_importances, ignore_index=True)
            
            # Average importance across models
            avg_importance = consolidated.groupby('feature')['importance'].agg(['mean', 'std', 'count']).reset_index()
            avg_importance = avg_importance.sort_values('mean', ascending=False)
            
            return avg_importance
        else:
            return pd.DataFrame()

# ================================
# ML/requirements.txt (Additional)
# ================================

"""
# Enhanced ML Pipeline Requirements
# Add these to your existing requirements.txt

# Core ML libraries
scikit-learn>=1.3.0
xgboost>=1.7.0
lightgbm>=3.3.0

# Time series
statsmodels>=0.14.0
prophet>=1.1.0

# Visualization
plotly>=5.15.0
dash>=2.10.0
seaborn>=0.12.0

# Data processing
pandas>=1.5.0
numpy>=1.24.0

# Model management
joblib>=1.3.0
mlflow>=2.5.0  # Optional for experiment tracking

# Performance
scipy>=1.10.0
"""

# ================================
# ML/config_example.py
# ================================

# Example configuration file showing how to customize the pipeline

CUSTOM_CONFIG = {
    'model_config': {
        'algorithms': {
            'regression': ['random_forest', 'xgboost', 'lightgbm', 'ridge'],
            'classification': ['random_forest', 'xgboost', 'logistic', 'svm'],
            'time_series': ['exponential_smoothing', 'arima', 'prophet']
        },
        'hyperparameter_tuning': {
            'enabled': True,
            'search_type': 'random',  # 'random' or 'grid'
            'n_iter': 20,
            'cv_folds': 5
        },
        'feature_engineering': {
            'auto_feature_selection': True,
            'max_features': 20,
            'polynomial_features': False,
            'interaction_terms': True
        }
    },
    'business_rules': {
        'profit_margin_threshold': 0.15,  # 15%
        'ontime_delivery_threshold': 0.95,  # 95%
        'customer_value_segments': 5,
        'forecast_confidence_level': 0.95
    },
    'monitoring': {
        'drift_detection_threshold': 0.05,
        'performance_alert_threshold': 0.1,
        'retraining_schedule': 'weekly',
        'data_quality_checks': True
    }
}

# Usage example:
# config = ModelConfig()
# config.update_from_dict(CUSTOM_CONFIG['model_config'])
