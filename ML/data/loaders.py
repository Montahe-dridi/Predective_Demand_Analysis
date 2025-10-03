# ================================
# ML/data/loaders.py
# ================================

import pandas as pd
import numpy as np
from typing import Optional, Dict
from ML.config.model_config import ModelConfig
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
