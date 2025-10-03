import pandas as pd
import os
import sys
import numpy as np
from typing import Dict, Any
from datetime import datetime
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
   
    mean_squared_error,
    r2_score,
   
)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ML.config.model_config import ModelConfig


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
def run_monitoring():
    """Entry point for monitoring pipeline"""
    print("🚀 Starting monitoring pipeline...")

    # Example usage: load reference & current data
    # In practice, replace these with real data loaders
    ref_data = pd.DataFrame({
        "feature1": np.random.normal(0, 1, 1000),
        "feature2": np.random.normal(5, 2, 1000)
    })
    curr_data = pd.DataFrame({
        "feature1": np.random.normal(0.2, 1, 1000),
        "feature2": np.random.normal(5.5, 2, 1000)
    })

    # Create monitoring pipeline
    from ML.config.model_config import ModelConfig
    config = ModelConfig()
    monitor = ModelMonitoringPipeline(config)

    # Run drift detection
    drift = monitor.detect_data_drift(ref_data, curr_data)
    print("📊 Drift detection results:")
    print(drift)

    # Example: monitor model performance (classification case)
    y_true = np.random.randint(0, 2, 500)
    y_pred = np.random.randint(0, 2, 500)
    performance = monitor.monitor_model_performance("demo_model", y_true, y_pred, task_type="classification")

    print("📈 Performance monitoring:")
    print(performance)

    if monitor.alerts:
        print("⚠️ Alerts:")
        for alert in monitor.alerts:
            print(alert)
      