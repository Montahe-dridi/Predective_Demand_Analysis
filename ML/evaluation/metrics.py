# ML/evaluation/metrics.py

import pandas as pd
import numpy as np
from sklearn.metrics import *
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ML.config.model_config import ModelConfig

class ModelEvaluator:
    """Comprehensive model evaluation and monitoring"""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
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
        
        # Fixed: Import max_error_score properly
        try:
            from sklearn.metrics import max_error
            max_err = max_error(y_true, y_pred)
        except ImportError:
            # Fallback calculation if max_error not available
            max_err = np.max(np.abs(y_true - y_pred))
        
        # Median Absolute Error
        median_ae = np.median(np.abs(y_true - y_pred))
        
        # Residual analysis
        residuals = y_true - y_pred
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        
        # Normality test for residuals
        try:
            _, normality_pvalue = stats.normaltest(residuals)
        except:
            normality_pvalue = 0.5  # Default if test fails
        
        # Heteroscedasticity test (simplified)
        try:
            abs_residuals = np.abs(residuals)
            corr_result = stats.spearmanr(y_pred, abs_residuals)
            hetero_pvalue = abs(corr_result[0]) if corr_result[1] is not None else 0.5
        except:
            hetero_pvalue = 0.5  # Default if test fails
        
        results = {
            'model_name': model_name,
            'rmse': rmse,
            'mae': mae,
            'median_ae': median_ae,
            'r2_score': r2,
            'mape': mape,
            'max_error': max_err,
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'residuals_normal': normality_pvalue > 0.05,
            'homoscedastic': hetero_pvalue < 0.3,
            'evaluation_timestamp': pd.Timestamp.now()
        }
        
        # Generate plots
        try:
            self._plot_regression_diagnostics(y_true, y_pred, model_name)
        except Exception as e:
            print(f"⚠️ Could not generate diagnostic plots: {e}")
        
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
        
        # Class-specific metrics
        try:
            class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        except:
            class_report = {}
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'evaluation_timestamp': pd.Timestamp.now()
        }
        
        # ROC AUC for binary classification
        if len(np.unique(y_true)) == 2 and y_pred_proba is not None:
            try:
                roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                results['roc_auc'] = roc_auc
                
                # Additional binary classification metrics
                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                
                results.update({
                    'specificity': specificity,
                    'npv': npv,
                    'true_positives': int(tp),
                    'true_negatives': int(tn),
                    'false_positives': int(fp),
                    'false_negatives': int(fn)
                })
                
            except Exception as e:
                print(f"⚠️ Error calculating ROC AUC: {e}")
        
        # Generate plots
        try:
            self._plot_classification_diagnostics(y_true, y_pred, y_pred_proba, model_name)
        except Exception as e:
            print(f"⚠️ Could not generate diagnostic plots: {e}")
        
        self.evaluation_history.append(results)
        return results
    
    def evaluate_time_series(self, y_true: pd.Series, y_pred: pd.Series,
                           model_name: str = "ts_model") -> Dict[str, Any]:
        """Time series specific evaluation"""
        
        # Convert to numpy arrays for calculations
        y_true_vals = y_true.values if hasattr(y_true, 'values') else y_true
        y_pred_vals = y_pred.values if hasattr(y_pred, 'values') else y_pred
        
        # Standard regression metrics
        rmse = np.sqrt(mean_squared_error(y_true_vals, y_pred_vals))
        mae = mean_absolute_error(y_true_vals, y_pred_vals)
        
        # Time series specific metrics
        mape = np.mean(np.abs((y_true_vals - y_pred_vals) / (y_true_vals + 1e-8))) * 100
        smape = np.mean(2 * np.abs(y_true_vals - y_pred_vals) / (np.abs(y_true_vals) + np.abs(y_pred_vals) + 1e-8)) * 100
        
        # Directional accuracy
        if len(y_true_vals) > 1:
            try:
                if hasattr(y_true, 'diff'):
                    direction_true = np.sign(y_true.diff().dropna())
                    direction_pred = np.sign(y_pred.diff().dropna())
                else:
                    direction_true = np.sign(np.diff(y_true_vals))
                    direction_pred = np.sign(np.diff(y_pred_vals))
                directional_accuracy = np.mean(direction_true == direction_pred) * 100
            except:
                directional_accuracy = np.nan
        else:
            directional_accuracy = np.nan
        
        # Bias and variance decomposition
        bias = np.mean(y_pred_vals - y_true_vals)
        variance = np.var(y_pred_vals - y_true_vals)
        
        results = {
            'model_name': model_name,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'smape': smape,
            'directional_accuracy': directional_accuracy,
            'bias': bias,
            'variance': variance,
            'evaluation_timestamp': pd.Timestamp.now()
        }
        
        # Generate plots
        try:
            self._plot_time_series_diagnostics(y_true, y_pred, model_name)
        except Exception as e:
            print(f"⚠️ Could not generate time series plots: {e}")
        
        return results
    
    def _plot_regression_diagnostics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   model_name: str):
        """Generate regression diagnostic plots"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Predicted vs Actual
            axes[0, 0].scatter(y_true, y_pred, alpha=0.6, color='blue')
            axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            axes[0, 0].set_xlabel('Actual')
            axes[0, 0].set_ylabel('Predicted')
            axes[0, 0].set_title('Predicted vs Actual')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add R² to the plot
            r2 = r2_score(y_true, y_pred)
            axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 0].transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Residuals vs Predicted
            residuals = y_true - y_pred
            axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
            axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
            axes[0, 1].set_xlabel('Predicted')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title('Residual Plot')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Residual histogram
            axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black', color='orange')
            axes[1, 0].set_xlabel('Residuals')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Residual Distribution')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add normality test result
            try:
                _, p_value = stats.normaltest(residuals)
                axes[1, 0].text(0.05, 0.95, f'Normality p-value: {p_value:.3f}', 
                               transform=axes[1, 0].transAxes,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            except:
                pass
            
            # Q-Q plot
            try:
                stats.probplot(residuals, dist="norm", plot=axes[1, 1])
                axes[1, 1].set_title('Q-Q Plot (Normality Check)')
                axes[1, 1].grid(True, alpha=0.3)
            except:
                axes[1, 1].text(0.5, 0.5, 'Q-Q Plot unavailable', ha='center', va='center', transform=axes[1, 1].transAxes)
            
            plt.suptitle(f'Regression Diagnostics: {model_name}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            plot_path = os.path.join(self.config.PLOTS_DIR, f"{model_name}_regression_diagnostics.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️ Error creating regression plots: {e}")
    
    def _plot_classification_diagnostics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                       y_pred_proba: np.ndarray, model_name: str):
        """Generate classification diagnostic plots"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
            axes[0, 0].set_title('Confusion Matrix')
            axes[0, 0].set_xlabel('Predicted')
            axes[0, 0].set_ylabel('Actual')
            
            # Class distribution
            unique, counts = np.unique(y_true, return_counts=True)
            axes[0, 1].bar(unique, counts, alpha=0.7, color='lightgreen')
            axes[0, 1].set_title('Class Distribution')
            axes[0, 1].set_xlabel('Class')
            axes[0, 1].set_ylabel('Count')
            
            # ROC Curve (for binary classification)
            if len(np.unique(y_true)) == 2 and y_pred_proba is not None:
                try:
                    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
                    roc_auc = auc(fpr, tpr)
                    axes[1, 0].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='darkorange', lw=2)
                    axes[1, 0].plot([0, 1], [0, 1], 'k--', lw=2)
                    axes[1, 0].set_xlabel('False Positive Rate')
                    axes[1, 0].set_ylabel('True Positive Rate')
                    axes[1, 0].set_title('ROC Curve')
                    axes[1, 0].legend()
                    axes[1, 0].grid(True, alpha=0.3)
                except Exception as e:
                    axes[1, 0].text(0.5, 0.5, f'ROC curve error: {str(e)[:50]}', ha='center', va='center', transform=axes[1, 0].transAxes)
            else:
                axes[1, 0].text(0.5, 0.5, 'ROC curve not applicable', ha='center', va='center', transform=axes[1, 0].transAxes)
            
            # Prediction confidence distribution
            if y_pred_proba is not None:
                try:
                    max_proba = np.max(y_pred_proba, axis=1)
                    axes[1, 1].hist(max_proba, bins=20, alpha=0.7, edgecolor='black', color='purple')
                    axes[1, 1].set_xlabel('Maximum Prediction Probability')
                    axes[1, 1].set_ylabel('Frequency')
                    axes[1, 1].set_title('Prediction Confidence')
                    axes[1, 1].grid(True, alpha=0.3)
                    
                    # Add confidence statistics
                    avg_confidence = np.mean(max_proba)
                    axes[1, 1].axvline(avg_confidence, color='red', linestyle='--', 
                                     label=f'Avg: {avg_confidence:.2f}')
                    axes[1, 1].legend()
                except Exception as e:
                    axes[1, 1].text(0.5, 0.5, f'Confidence plot error', ha='center', va='center', transform=axes[1, 1].transAxes)
            else:
                axes[1, 1].text(0.5, 0.5, 'Probabilities not available', ha='center', va='center', transform=axes[1, 1].transAxes)
            
            plt.suptitle(f'Classification Diagnostics: {model_name}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            plot_path = os.path.join(self.config.PLOTS_DIR, f"{model_name}_classification_diagnostics.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️ Error creating classification plots: {e}")
    
    def _plot_time_series_diagnostics(self, y_true: pd.Series, y_pred: pd.Series, 
                                    model_name: str):
        """Generate time series diagnostic plots"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Time series plot
            if hasattr(y_true, 'index') and hasattr(y_pred, 'index'):
                axes[0, 0].plot(y_true.index, y_true.values, label='Actual', alpha=0.8, color='blue', linewidth=2)
                axes[0, 0].plot(y_pred.index, y_pred.values, label='Predicted', alpha=0.8, color='red', linewidth=2)
            else:
                axes[0, 0].plot(y_true, label='Actual', alpha=0.8, color='blue', linewidth=2)
                axes[0, 0].plot(y_pred, label='Predicted', alpha=0.8, color='red', linewidth=2)
            
            axes[0, 0].set_title('Time Series: Actual vs Predicted')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Residuals over time
            residuals = y_true - y_pred if hasattr(y_true, '__sub__') else np.array(y_true) - np.array(y_pred)
            if hasattr(y_true, 'index'):
                axes[0, 1].plot(y_true.index, residuals, alpha=0.7, color='green')
            else:
                axes[0, 1].plot(residuals, alpha=0.7, color='green')
            axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
            axes[0, 1].set_title('Residuals Over Time')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Error distribution
            errors = np.abs(residuals)
            axes[1, 1].hist(errors, bins=20, alpha=0.7, edgecolor='black', color='coral')
            axes[1, 1].set_xlabel('Absolute Error')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Error Distribution')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add error statistics
            mean_error = np.mean(errors)
            axes[1, 1].axvline(mean_error, color='red', linestyle='--', 
                             label=f'Mean: {mean_error:.2f}')
            axes[1, 1].legend()
            
            # ACF of residuals or residual distribution
            try:
                from statsmodels.tsa.stattools import acf
                from statsmodels.graphics.tsaplots import plot_acf
                plot_acf(residuals.dropna() if hasattr(residuals, 'dropna') else residuals, ax=axes[1, 0], lags=min(20, len(residuals)//4))
                axes[1, 0].set_title('Residual Autocorrelation')
            except Exception:
                axes[1, 0].hist(residuals, bins=20, alpha=0.7, color='lightblue')
                axes[1, 0].set_title('Residual Distribution')
                axes[1, 0].grid(True, alpha=0.3)
            
            plt.suptitle(f'Time Series Diagnostics: {model_name}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            plot_path = os.path.join(self.config.PLOTS_DIR, f"{model_name}_ts_diagnostics.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️ Error creating time series plots: {e}")
    
    def compare_models(self, evaluation_results: List[Dict]) -> pd.DataFrame:
        """Compare multiple model results"""
        
        comparison_data = []
        
        for result in evaluation_results:
            model_data = {
                'Model': result['model_name'],
                'Timestamp': result['evaluation_timestamp']
            }
            
            # Add regression metrics if available
            if 'rmse' in result:
                model_data.update({
                    'RMSE': result['rmse'],
                    'MAE': result['mae'],
                    'R²': result['r2_score'],
                    'MAPE': result['mape']
                })
            
            # Add classification metrics if available
            if 'accuracy' in result:
                model_data.update({
                    'Accuracy': result['accuracy'],
                    'Precision': result['precision'],
                    'Recall': result['recall'],
                    'F1_Score': result['f1_score']
                })
                
                if 'roc_auc' in result:
                    model_data['ROC_AUC'] = result['roc_auc']
            
            comparison_data.append(model_data)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Rank models
        if 'R²' in comparison_df.columns:
            comparison_df['R²_Rank'] = comparison_df['R²'].rank(ascending=False)
        if 'Accuracy' in comparison_df.columns:
            comparison_df['Accuracy_Rank'] = comparison_df['Accuracy'].rank(ascending=False)
        
        return comparison_df
    
    def export_evaluation_report(self, output_path: str = None) -> str:
        """Export comprehensive evaluation report"""
        if output_path is None:
            output_path = os.path.join(self.config.REPORTS_DIR, "evaluation_report.json")
        
        report_data = {
            'evaluation_summary': {
                'total_evaluations': len(self.evaluation_history),
                'report_timestamp': pd.Timestamp.now().isoformat()
            },
            'evaluations': self.evaluation_history
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"📊 Evaluation report saved: {output_path}")
        return output_path