# ML/data/preprocessors.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ML.config.model_config import ModelConfig

class DataPreprocessor:
    """Advanced data preprocessing for ML pipeline"""
    
    def __init__(self):
        self.config = ModelConfig()
        self.scalers = {}
        self.preprocessing_stats = {}
    
    def preprocess_shipments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced shipment data preprocessing"""
        df = df.copy()
        
        print("🔧 Preprocessing shipment data...")
        
        # Date preprocessing
        date_cols = ['ShipmentDate', 'PlannedArrivalDate', 'ActualArrivalDate', 'ActualDepartureDate']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                # Handle 1970 dates
                mask_1970 = df[col].dt.year == 1970
                if mask_1970.any():
                 df.loc[mask_1970, col] = pd.NaT
                # Log date parsing issues
                null_dates = df[col].isnull().sum()
                if null_dates > 0:
                    print(f"⚠️ {null_dates} invalid dates in {col}")
        
        # Handle delivery performance metrics
        if all(col in df.columns for col in ['PlannedArrivalDate', 'ActualArrivalDate']):
            df['DeliveryVariance'] = (df['ActualArrivalDate'] - df['PlannedArrivalDate']).dt.days
            df['OnTimeDeliveryFlag'] = (df['DeliveryVariance'] <= 0).astype(int)
        elif 'OnTimeDeliveryFlag' in df.columns:
            # Clean existing OnTimeDeliveryFlag
            df['OnTimeDeliveryFlag'] = df['OnTimeDeliveryFlag'].replace({
                'Yes': 1, 'No': 0, 'True': 1, 'False': 0, True: 1, False: 0
            })
            df['OnTimeDeliveryFlag'] = pd.to_numeric(df['OnTimeDeliveryFlag'], errors='coerce').fillna(0).astype(int)
        
        # Outlier detection and handling
        df = self._handle_outliers(df, ['TotalWeight', 'TotalVolume', 'ShipmentValue'])
        
        # Missing value imputation
        df = self._impute_missing_values(df)
        
        # Data type optimization
        df = self._optimize_data_types(df)
        
        # Save preprocessing statistics
        self.preprocessing_stats['shipments'] = self._calculate_preprocessing_stats(df)
        
        print(f"✅ Shipment preprocessing complete: {len(df)} records")
        return df
    
    def preprocess_invoices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced invoice data preprocessing"""
        df = df.copy()
        
        print("🔧 Preprocessing invoice data...")
        
        # Date preprocessing
        date_cols = ['InvoiceDate', 'PaymentDueDate']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Financial data cleaning
        amount_cols = ['TotalAmount', 'NetAmount', 'TaxAmount']
        for col in amount_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                # Remove negative amounts (likely data errors)
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    print(f"⚠️ Found {negative_count} negative values in {col}, setting to 0")
                    df[col] = df[col].clip(lower=0)
        
        # Payment status normalization
        if 'PaymentStatus' in df.columns:
            df['PaymentStatus'] = self._normalize_payment_status(df['PaymentStatus'])
        
        # Create profit estimation
        if 'NetAmount' in df.columns and 'InvoiceProfit' not in df.columns:
            # Simple profit estimation (30% margin)
            cost_ratio = float(os.environ.get('INVOICE_COST_RATIO', 0.7))
            df['EstimatedCost'] = df['NetAmount'] * cost_ratio
            df['InvoiceProfit'] = df['NetAmount'] - df['EstimatedCost']
        
        # Tax rate calculation
        if all(col in df.columns for col in ['TaxAmount', 'NetAmount']):
            df['TaxRate'] = df['TaxAmount'] / (df['NetAmount'] + 1e-6)
            df['TaxRate'] = df['TaxRate'].clip(0, 1)  # Cap at 100%
        
        # Outlier handling
        df = self._handle_outliers(df, ['TotalAmount', 'NetAmount', 'TaxAmount'])
        
        # Missing value imputation
        df = self._impute_missing_values(df)
        
        # Data type optimization
        df = self._optimize_data_types(df)
        
        self.preprocessing_stats['invoices'] = self._calculate_preprocessing_stats(df)
        
        print(f"✅ Invoice preprocessing complete: {len(df)} records")
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, columns: List[str], 
                        method: str = 'iqr', factor: float = 1.5) -> pd.DataFrame:
        """Handle outliers using IQR method"""
        df = df.copy()
        outlier_counts = {}
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_counts[col] = outliers.sum()
                
                # Cap outliers instead of removing
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        if outlier_counts:
            total_outliers = sum(outlier_counts.values())
            print(f"🔧 Handled {total_outliers} outliers across {len(outlier_counts)} columns")
        
        return df
    
    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Intelligent missing value imputation"""
        df = df.copy()
        
        # Numeric columns - use median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        # Categorical columns - use mode or 'Unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                if df[col].notna().any():
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                    df[col] = df[col].fillna(mode_val)
                else:
                    df[col] = df[col].fillna('Unknown')
        
        # Boolean columns
        bool_cols = df.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(False)
        
        return df
    
    def _normalize_payment_status(self, payment_series: pd.Series) -> pd.Series:
        """Normalize payment status to consistent integer codes"""
        
        # Standard mapping
        status_mapping = {
            'Validée': 1, 'Payée': 2, 'En attente': 3, 'Annulée': 4,
            'Paid': 2, 'Unpaid': 0, 'Pending': 3, 'Cancelled': 4,
            'Validated': 1, 'Partially Paid': 3, 'Overdue': 5
        }
        
        # Apply mapping
        normalized = payment_series.map(status_mapping)
        
        # Handle unmapped values
        unmapped_mask = normalized.isnull() & payment_series.notna()
        if unmapped_mask.any():
            # Try to convert numeric values directly
            numeric_conversion = pd.to_numeric(payment_series[unmapped_mask], errors='coerce')
            normalized[unmapped_mask] = numeric_conversion
            
            # Any remaining unmapped values get code 0 (unknown)
            normalized = normalized.fillna(0)
        
        return normalized.astype(int)
    
    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency"""
        df = df.copy()
        
        # Integer optimization
        int_cols = df.select_dtypes(include=['int64']).columns
        for col in int_cols:
            if df[col].min() >= 0:
                if df[col].max() <= 255:
                    df[col] = df[col].astype('uint8')
                elif df[col].max() <= 65535:
                    df[col] = df[col].astype('uint16')
                elif df[col].max() <= 4294967295:
                    df[col] = df[col].astype('uint32')
            else:
                if df[col].min() >= -128 and df[col].max() <= 127:
                    df[col] = df[col].astype('int8')
                elif df[col].min() >= -32768 and df[col].max() <= 32767:
                    df[col] = df[col].astype('int16')
                elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                    df[col] = df[col].astype('int32')
        
        # Float optimization
        float_cols = df.select_dtypes(include=['float64']).columns
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        return df
    
    def _calculate_preprocessing_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate preprocessing statistics for monitoring"""
        
        return {
            'total_records': len(df),
            'total_features': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object']).columns),
            'datetime_features': len(df.select_dtypes(include=['datetime64']).columns)
        }