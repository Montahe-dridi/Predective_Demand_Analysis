# ML/evaluation/monitoring.py

import pandas as pd
import numpy as np
from typing import Dict, Any,List
from scipy import stats
from datetime import datetime
import json
import sys
import os
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
    r2_score,
    mean_absolute_error
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ML.config.model_config import ModelConfig

class ModelMonitoringPipeline:
    """Model performance monitoring and drift detection"""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.baseline_stats = {}
        self.alerts = []
        self.drift_history = []
    
    def detect_data_drift(self, reference_data: pd.DataFrame, 
                         current_data: pd.DataFrame,
                         threshold: float = 0.05) -> Dict[str, Any]:
        """Detect data drift using statistical tests"""
        
        drift_results = {}
        total_drift_score = 0
        
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
                    try:
                        mw_stat, mw_pvalue = stats.mannwhitneyu(ref_values, curr_values, 
                                                               alternative='two-sided')
                    except ValueError:
                        mw_stat, mw_pvalue = np.nan, 1.0
                    
                    # Population Stability Index (PSI)
                    psi_score = self._calculate_psi(ref_values, curr_values)
                    
                    drift_detected = ks_pvalue < threshold or mw_pvalue < threshold or psi_score > 0.2
                    
                    drift_results[col] = {
                        'drift_detected': drift_detected,
                        'ks_statistic': ks_stat,
                        'ks_pvalue': ks_pvalue,
                        'mw_pvalue': mw_pvalue,
                        'psi_score': psi_score,
                        'ref_mean': ref_values.mean(),
                        'curr_mean': curr_values.mean(),
                        'mean_shift': abs(curr_values.mean() - ref_values.mean()) / (ref_values.std() + 1e-8),
                        'ref_std': ref_values.std(),
                        'curr_std': curr_values.std()
                    }
                    
                    if drift_detected:
                        self.alerts.append(f"⚠️ Data drift detected in {col}")
                        total_drift_score += 1
        
        drift_summary = {
            'drift_results': drift_results,
            'total_features_checked': len(numeric_cols),
            'features_with_drift': sum(1 for r in drift_results.values() if r['drift_detected']),
            'overall_drift_score': total_drift_score / len(numeric_cols) if numeric_cols else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        self.drift_history.append(drift_summary)
        return drift_summary
    
    def _calculate_psi(self, reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """Calculate Population Stability Index"""
        try:
            # Create bins based on reference data
            _, bin_edges = pd.cut(reference, bins=bins, retbins=True)
            
            # Calculate distributions
            ref_dist = pd.cut(reference, bins=bin_edges, include_lowest=True).value_counts(normalize=True, sort=False)
            curr_dist = pd.cut(current, bins=bin_edges, include_lowest=True).value_counts(normalize=True, sort=False)
            
            # Add small epsilon to avoid log(0)
            ref_dist = ref_dist.fillna(0) + 1e-10
            curr_dist = curr_dist.fillna(0) + 1e-10
            
            # Calculate PSI
            psi = np.sum((curr_dist - ref_dist) * np.log(curr_dist / ref_dist))
            return psi
            
        except Exception as e:
            print(f"⚠️ Error calculating PSI: {e}")
            return 0.0
    
    def monitor_model_performance(self, model_name: str, y_true: np.ndarray, 
                                y_pred: np.ndarray, task_type: str = 'regression') -> Dict:
        """Monitor model performance over time"""
        
        current_timestamp = datetime.now()
        
        if task_type == 'regression':
            current_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            current_r2 = r2_score(y_true, y_pred)
            current_mae = mean_absolute_error(y_true, y_pred)
            
            current_metrics = {
                'rmse': current_rmse,
                'r2': current_r2,
                'mae': current_mae,
                'timestamp': current_timestamp
            }
            
            # Check against baseline
            if model_name in self.baseline_stats:
                baseline = self.baseline_stats[model_name]
                
                rmse_change = (current_rmse - baseline['rmse']) / baseline['rmse']
                r2_change = (baseline['r2'] - current_r2) / baseline['r2']
                
                if rmse_change > 0.1:  # 10% worse
                    self.alerts.append(f"⚠️ {model_name} RMSE degraded by {rmse_change:.1%}")
                
                if r2_change > 0.1:
                    self.alerts.append(f"⚠️ {model_name} R² degraded by {r2_change:.1%}")
                
                current_metrics.update({
                    'baseline_rmse': baseline['rmse'],
                    'baseline_r2': baseline['r2'],
                    'rmse_change': rmse_change,
                    'r2_change': r2_change
                })
            else:
                # Set baseline
                self.baseline_stats[model_name] = current_metrics.copy()
            
            return current_metrics
        
        else:  # classification
            current_accuracy = accuracy_score(y_true, y_pred)
            current_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            current_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            current_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            
            current_metrics = {
                'accuracy': current_accuracy,
                'f1_score': current_f1,
                'precision': current_precision,
                'recall': current_recall,
                'timestamp': current_timestamp
            }
            
            if model_name in self.baseline_stats:
                baseline = self.baseline_stats[model_name]
                acc_change = (baseline['accuracy'] - current_accuracy) / baseline['accuracy']
                
                if acc_change > 0.05:  # 5% worse
                    self.alerts.append(f"⚠️ {model_name} accuracy degraded by {acc_change:.1%}")
                
                current_metrics.update({
                    'baseline_accuracy': baseline['accuracy'],
                    'accuracy_change': acc_change
                })
            else:
                self.baseline_stats[model_name] = current_metrics.copy()
            
            return current_metrics
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        
        report = {
            'monitoring_summary': {
                'total_alerts': len(self.alerts),
                'models_monitored': len(self.baseline_stats),
                'drift_checks_performed': len(self.drift_history),
                'report_timestamp': datetime.now().isoformat()
            },
            'active_alerts': self.alerts,
            'baseline_statistics': self.baseline_stats,
            'drift_history': self.drift_history,
            'recommendations': self._generate_monitoring_recommendations()
        }
        
        # Save report
        report_path = os.path.join(self.config.REPORTS_DIR, "monitoring_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Monitoring report saved: {report_path}")
        return report
    
    def _generate_monitoring_recommendations(self) -> List[str]:
        """Generate monitoring-based recommendations"""
        recommendations = []
        
        # Alert-based recommendations
        if len(self.alerts) > 5:
            recommendations.append("🚨 High number of alerts detected. Consider immediate model retraining.")
        elif len(self.alerts) > 0:
            recommendations.append("⚠️ Some performance issues detected. Monitor closely.")
        else:
            recommendations.append("✅ All models performing within acceptable ranges.")
        
        # Drift-based recommendations
        if self.drift_history:
            latest_drift = self.drift_history[-1]
            if latest_drift['overall_drift_score'] > 0.3:
                recommendations.append("📊 Significant data drift detected. Update feature engineering and retrain models.")
            elif latest_drift['overall_drift_score'] > 0.1:
                recommendations.append("📈 Moderate data drift detected. Schedule model retraining.")
        
        # Performance trend recommendations
        if len(self.baseline_stats) > 0:
            recommendations.append("🔄 Establish regular monitoring schedule (weekly/monthly) for production models.")
        
        return recommendations
    
    def clear_alerts(self):
        """Clear current alerts after review"""
        cleared_count = len(self.alerts)
        self.alerts = []
        print(f"✅ Cleared {cleared_count} alerts")
    
    def get_alert_summary(self) -> Dict[str, int]:
        """Get summary of current alerts"""
        return {
            'total_alerts': len(self.alerts),
            'drift_alerts': len([a for a in self.alerts if 'drift' in a.lower()]),
            'performance_alerts': len([a for a in self.alerts if 'degraded' in a.lower()])
        }