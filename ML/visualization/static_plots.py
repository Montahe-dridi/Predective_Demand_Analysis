# ML/visualization/static_plots.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ML.config.model_config import ModelConfig

class StaticPlotGenerator:
    """Generate high-quality static plots for reports and presentations"""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        
        # Set styling
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Configure matplotlib
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
    
    def plot_model_comparison(self, model_results: Dict[str, Dict], 
                            metric: str = 'auto', save_path: str = None) -> str:
        """Create model comparison plot"""
        
        # Extract data for plotting
        models = []
        scores = []
        algorithms = []
        
        for model_name, results in model_results.items():
            if 'evaluation' in results:
                eval_metrics = results['evaluation']
                
                # Auto-detect metric
                if metric == 'auto':
                    if 'r2_score' in eval_metrics:
                        metric = 'r2_score'
                    elif 'accuracy' in eval_metrics:
                        metric = 'accuracy'
                    else:
                        continue
                
                if metric in eval_metrics:
                    models.append(model_name)
                    scores.append(eval_metrics[metric])
                    algorithms.append(results.get('algorithm', model_name))
        
        if not models:
            print("⚠️ No model results found for comparison")
            return None
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create bar plot
        bars = ax.bar(models, scores, color=sns.color_palette("husl", len(models)))
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Styling
        ax.set_title(f'Model Performance Comparison - {metric.upper()}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=14)
        ax.set_xlabel('Models', fontsize=14)
        
        # Rotate x-axis labels if too many models
        if len(models) > 5:
            plt.xticks(rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add benchmark line if applicable
        if metric == 'r2_score':
            ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='Minimum Target (0.6)')
        elif metric == 'accuracy':
            ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Minimum Target (0.7)')
        
        if 'Minimum Target' in [t.get_text() for t in ax.get_legend_handles_labels()[1]]:
            ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.config.PLOTS_DIR, f"model_comparison_{metric}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"💾 Model comparison plot saved: {save_path}")
        return save_path
    
    def plot_feature_importance(self, feature_importance_df: pd.DataFrame,
                              title: str = "Feature Importance",
                              top_n: int = 15, save_path: str = None) -> str:
        """Create feature importance plot"""
        
        if feature_importance_df.empty:
            print("⚠️ No feature importance data available")
            return None
        
        # Take top N features
        top_features = feature_importance_df.head(top_n)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, max(8, len(top_features) * 0.4)))
        
        # Horizontal bar plot
        bars = ax.barh(range(len(top_features)), top_features['importance'])
        
        # Color bars by importance (gradient)
        colors = plt.cm.viridis(top_features['importance'] / top_features['importance'].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Styling
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance Score', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
            width = bar.get_width()
            ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                   f'{importance:.3f}', ha='left', va='center', fontsize=10)
        
        # Invert y-axis to show highest importance at top
        ax.invert_yaxis()
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.config.PLOTS_DIR, f"feature_importance_{title.lower().replace(' ', '_')}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"💾 Feature importance plot saved: {save_path}")
        return save_path
    
    def plot_customer_segments(self, segmented_data: pd.DataFrame,
                             save_path: str = None) -> str:
        """Create customer segment visualization"""
        
        if 'Segment_Business' not in segmented_data.columns:
            print("⚠️ No business segments found in data")
            return None
        
        # Create subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Segment size distribution
        segment_counts = segmented_data['Segment_Business'].value_counts()
        axes[0, 0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Customer Segment Distribution', fontweight='bold')
        
        # 2. RFM by segment (box plots)
        if all(col in segmented_data.columns for col in ['Recency', 'Frequency', 'Monetary']):
            # Monetary by segment
            segmented_data.boxplot(column='Monetary', by='Segment_Business', ax=axes[0, 1])
            axes[0, 1].set_title('Monetary Value by Segment')
            axes[0, 1].set_xlabel('Customer Segment')
            axes[0, 1].set_ylabel('Monetary Value ($)')
            plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Frequency vs Recency scatter
        if all(col in segmented_data.columns for col in ['Recency', 'Frequency']):
            for i, segment in enumerate(segmented_data['Segment_Business'].unique()):
                segment_data = segmented_data[segmented_data['Segment_Business'] == segment]
                axes[1, 0].scatter(segment_data['Recency'], segment_data['Frequency'], 
                                 label=segment, alpha=0.6, s=30)
            
            axes[1, 0].set_xlabel('Recency (Days)')
            axes[1, 0].set_ylabel('Frequency (Transactions)')
            axes[1, 0].set_title('Recency vs Frequency by Segment')
            axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Segment profitability
        if 'TotalRevenue' in segmented_data.columns:
            segment_revenue = segmented_data.groupby('Segment_Business')['TotalRevenue'].sum()
            axes[1, 1].bar(segment_revenue.index, segment_revenue.values)
            axes[1, 1].set_title('Total Revenue by Segment')
            axes[1, 1].set_ylabel('Total Revenue ($)')
            plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.suptitle('Customer Segmentation Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.config.PLOTS_DIR, "customer_segmentation_analysis.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"💾 Customer segmentation plot saved: {save_path}")
        return save_path
    
    def plot_time_series_forecast(self, historical: pd.Series, forecast: pd.Series,
                                confidence_interval: Tuple[pd.Series, pd.Series] = None,
                                title: str = "Time Series Forecast",
                                save_path: str = None) -> str:
        """Create time series forecast plot"""
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Historical data
        ax.plot(historical.index, historical.values, 
               label='Historical', color='blue', linewidth=2, alpha=0.8)
        
        # Forecast
        ax.plot(forecast.index, forecast.values, 
               label='Forecast', color='red', linewidth=2, linestyle='--', alpha=0.8)
        
        # Confidence interval
        if confidence_interval:
            lower, upper = confidence_interval
            ax.fill_between(forecast.index, lower.values, upper.values,
                          alpha=0.3, color='red', label='95% Confidence Interval')
        
        # Styling
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Value', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add vertical line to separate historical from forecast
        if len(historical) > 0 and len(forecast) > 0:
            ax.axvline(x=historical.index[-1], color='gray', linestyle=':', alpha=0.7,
                      label='Forecast Start')
        
        # Format dates on x-axis
        ax.tick_params(axis='x', rotation=45)
        
        # Add statistics box
        stats_text = f"Historical Mean: {historical.mean():.2f}\n"
        stats_text += f"Forecast Mean: {forecast.mean():.2f}\n"
        stats_text += f"Forecast Period: {len(forecast)} days"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.config.PLOTS_DIR, f"forecast_{title.lower().replace(' ', '_')}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"💾 Forecast plot saved: {save_path}")
        return save_path
    
    def plot_business_metrics_summary(self, shipments: pd.DataFrame, 
                                    invoices: pd.DataFrame,
                                    save_path: str = None) -> str:
        """Create comprehensive business metrics summary plot"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Monthly shipment volume trend
        if 'ShipmentDate' in shipments.columns:
            shipments['ShipmentDate'] = pd.to_datetime(shipments['ShipmentDate'], errors='coerce')
            monthly_shipments = shipments.groupby(shipments['ShipmentDate'].dt.to_period('M')).size()
            
            axes[0, 0].plot(monthly_shipments.index.astype(str), monthly_shipments.values, 
                          marker='o', linewidth=2, markersize=6)
            axes[0, 0].set_title('Monthly Shipment Volume Trend', fontweight='bold')
            axes[0, 0].set_ylabel('Number of Shipments')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Revenue distribution
        if 'NetAmount' in invoices.columns:
            invoices['NetAmount'] = pd.to_numeric(invoices['NetAmount'], errors='coerce')
            axes[0, 1].hist(invoices['NetAmount'], bins=30, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Invoice Amount Distribution', fontweight='bold')
            axes[0, 1].set_xlabel('Invoice Amount ($)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(invoices['NetAmount'].mean(), color='red', linestyle='--', 
                             label=f'Mean: ${invoices["NetAmount"].mean():,.0f}')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. On-time delivery performance
        if 'OnTimeDeliveryFlag' in shipments.columns:
            ontime_rate = shipments['OnTimeDeliveryFlag'].mean() * 100
            categories = ['On-Time', 'Delayed']
            values = [ontime_rate, 100 - ontime_rate]
            colors = ['green', 'red']
            
            wedges, texts, autotexts = axes[0, 2].pie(values, labels=categories, autopct='%1.1f%%', 
                                                    colors=colors, startangle=90)
            axes[0, 2].set_title('Delivery Performance', fontweight='bold')
        
        # 4. Payment status distribution
        if 'PaymentStatus' in invoices.columns:
            payment_counts = invoices['PaymentStatus'].value_counts()
            status_labels = ['Unpaid', 'Validated', 'Paid', 'Pending', 'Cancelled']
            
            # Map numeric codes to labels
            display_data = []
            display_labels = []
            for status_code in payment_counts.index:
                if status_code < len(status_labels):
                    display_labels.append(status_labels[status_code])
                else:
                    display_labels.append(f'Status_{status_code}')
                display_data.append(payment_counts[status_code])
            
            axes[1, 0].bar(display_labels, display_data, alpha=0.7)
            axes[1, 0].set_title('Payment Status Distribution', fontweight='bold')
            axes[1, 0].set_ylabel('Number of Invoices')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Shipment value vs weight correlation
        if all(col in shipments.columns for col in ['ShipmentValue', 'TotalWeight']):
            axes[1, 1].scatter(shipments['TotalWeight'], shipments['ShipmentValue'], 
                             alpha=0.6, s=20)
            axes[1, 1].set_title('Shipment Value vs Weight', fontweight='bold')
            axes[1, 1].set_xlabel('Total Weight (kg)')
            axes[1, 1].set_ylabel('Shipment Value ($)')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add correlation coefficient
            corr = shipments[['TotalWeight', 'ShipmentValue']].corr().iloc[0, 1]
            axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                           transform=axes[1, 1].transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 6. Monthly revenue trend
        if 'InvoiceDate' in invoices.columns and 'NetAmount' in invoices.columns:
            invoices['InvoiceDate'] = pd.to_datetime(invoices['InvoiceDate'], errors='coerce')
            monthly_revenue = invoices.groupby(invoices['InvoiceDate'].dt.to_period('M'))['NetAmount'].sum()