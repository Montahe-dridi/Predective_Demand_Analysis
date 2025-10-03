# ================================
# ML/saved_models/main_enhanced_pipeline.py
# ================================

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, Any, Optional
import os
import sys

# Add ML directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(ml_dir)
sys.path.extend([ml_dir, project_dir])

from ML.config.model_config import ModelConfig
from ML.data.loaders import DataLoader
from ML.pipelines.training_pipeline import MLTrainingPipeline
from ML.utils.seasonal_kpi_exporter import SeasonalKPIExporter
from ML.features.engineering import FeatureEngineer


class EnhancedMLPipeline:
    """Enhanced ML pipeline with seasonal KPIs, predictions, and business reports"""
    
    def __init__(self, sample_limit: Optional[int] = None):
        self.config = ModelConfig()
        self.sample_limit = sample_limit
        self.results = {}
        self.data_loader = DataLoader()
        
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete enhanced ML pipeline with all exporters"""
        
        print("🚀 Starting Enhanced ML Pipeline for TRALIS Analytics")
        print("=" * 60)
        
        try:
            # 1. Data Loading
            print("\n📥 STEP 1: Enhanced Data Loading")
            shipments = self.data_loader.load_shipments(self.sample_limit)
            invoices = self.data_loader.load_invoices(self.sample_limit)
            
            if shipments.empty or invoices.empty:
                raise ValueError("No data loaded - check database connection")
            
            data_summary = self.data_loader.get_data_summary()
            print("✅ Data loaded successfully")
            for key, summary in data_summary.items():
                print(f"  🔹 {key}: {summary['rows']:,} rows, {summary['columns']} columns")
            
            # 2. Feature Engineering
            print("\n🔧 STEP 2: Advanced Feature Engineering")
            fe = FeatureEngineer()
            shipments = fe.engineer_shipment_features(shipments)
            invoices = fe.engineer_invoice_features(invoices)
            
            training_pipeline = MLTrainingPipeline(self.config)

            # 3. Regression Models
            print("\n🤖 STEP 3: Regression Models")
            
            # Delivery Variance
            try:
                delivery_results = training_pipeline.run_regression_pipeline(
                    shipments, 'DeliveryVariance',
                    algorithms=['random_forest', 'gradient_boost', 'ridge']
                )
                self.results['delivery_variance_models'] = delivery_results
            except Exception as e:
                print(f"❌ Delivery variance pipeline failed: {e}")
            
            # Invoice Profit
            try:
                profit_results = training_pipeline.run_regression_pipeline(
                    invoices, 'InvoiceProfit',
                    algorithms=['random_forest', 'gradient_boost', 'lasso']
                )
                self.results['invoice_profit_models'] = profit_results
            except Exception as e:
                print(f"❌ Invoice profit pipeline failed: {e}")
            
            # 4. Classification Models
            print("\n🧠 STEP 4: Classification Models")
            
            # On-time Delivery
            try:
                if 'OnTimeDeliveryFlag' in shipments.columns:
                    ontime_results = training_pipeline.run_classification_pipeline(
                        shipments, 'OnTimeDeliveryFlag',
                        algorithms=['random_forest', 'gradient_boost', 'logistic']
                    )
                    self.results['ontime_delivery_models'] = ontime_results
            except Exception as e:
                print(f"❌ On-time delivery classification failed: {e}")
            
            # Payment Status
            try:
                if 'PaymentStatus' in invoices.columns:
                    payment_results = training_pipeline.run_classification_pipeline(
                        invoices, 'PaymentStatus',
                        algorithms=['random_forest', 'gradient_boost', 'logistic']
                    )
                    self.results['payment_status_models'] = payment_results
            except Exception as e:
                print(f"❌ Payment status classification failed: {e}")

            # 4.5 Export Results
            print("\n📊 STEP 4.5: Export Results")

            # Seasonal KPIs
            try:
                exporter = SeasonalKPIExporter()
                exporter.export(shipments, invoices)
            except Exception as e:
                print(f"⚠️ Seasonal KPI export failed: {e}")

            
            # 5. Generate Summary Report
            print("\n📋 STEP 5: Generating Summary Report")
            final_report = self._generate_summary_report()
            
            print("\n🎉 Enhanced ML Pipeline Completed!")
            print("=" * 60)
            print(f"📊 Models trained: {self._count_trained_models()}")
            print(f"💾 Results saved to: {self.config.OUTPUT_DIR}")
            
            return final_report
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "status": "failed"}
    
    def _count_trained_models(self) -> int:
        count = 0
        for key, value in self.results.items():
            if isinstance(value, dict):
                for model_name, model_data in value.items():
                    if isinstance(model_data, dict) and 'evaluation' in model_data:
                        count += 1
        return count
    
    def _generate_summary_report(self) -> Dict[str, Any]:
        report = {
            'pipeline_info': {
                'execution_timestamp': datetime.now().isoformat(),
                'pipeline_version': '2.3_with_exports',
                'data_sample_limit': self.sample_limit,
                'total_models_trained': self._count_trained_models()
            },
            'data_summary': self.data_loader.get_data_summary(),
            'model_results': {},
            'recommendations': []
        }
        
        for pipeline_name, pipeline_results in self.results.items():
            if isinstance(pipeline_results, dict) and 'best_model' in pipeline_results:
                best_model = pipeline_results['best_model']
                if best_model and best_model in pipeline_results.get('all_results', {}):
                    best_results = pipeline_results['all_results'][best_model]['evaluation']
                    report['model_results'][pipeline_name] = {
                        'best_algorithm': best_model,
                        'r2_score': best_results.get('r2_score', 'N/A'),
                        'rmse': best_results.get('rmse', 'N/A'),
                        'mae': best_results.get('mae', 'N/A'),
                        'accuracy': best_results.get('accuracy', 'N/A'),
                        'feature_count': len(pipeline_results.get('selected_features', []))
                    }
        
        report['recommendations'] = self._generate_recommendations()
        
        report_path = os.path.join(self.config.REPORTS_DIR, "ml_pipeline_report.json")
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"📊 Report saved: {report_path}")
        except Exception as e:
            print(f"⚠️ Could not save report: {e}")
        
        return report
    
    def _generate_recommendations(self) -> list:
        recommendations = []
        for pipeline_name, pipeline_results in self.results.items():
            if isinstance(pipeline_results, dict) and 'best_model' in pipeline_results:
                best_model = pipeline_results['best_model']
                if best_model and best_model in pipeline_results.get('all_results', {}):
                    best_results = pipeline_results['all_results'][best_model]['evaluation']
                    if 'r2_score' in best_results:
                        score = best_results['r2_score']
                        if score > 0.8:
                            recommendations.append(f"✅ {pipeline_name}: Excellent performance (R² = {score:.3f})")
                        elif score > 0.6:
                            recommendations.append(f"🔶 {pipeline_name}: Good performance (R² = {score:.3f})")
                        else:
                            recommendations.append(f"⚠️ {pipeline_name}: Low performance (R² = {score:.3f})")
                    elif 'accuracy' in best_results:
                        accuracy = best_results['accuracy']
                        if accuracy > 0.8:
                            recommendations.append(f"✅ {pipeline_name}: Good classification (Accuracy = {accuracy:.3f})")
                        else:
                            recommendations.append(f"⚠️ {pipeline_name}: Weak classification (Accuracy = {accuracy:.3f})")
        if not recommendations:
            recommendations.append("🔄 Pipeline completed but no strong insights. Try more data/features.")
        return recommendations


def run_enhanced_ml_pipeline(sample_limit: int = None):
    try:
        pipeline = EnhancedMLPipeline(sample_limit=sample_limit)
        results = pipeline.run_complete_pipeline()
        return results
    except Exception as e:
        print(f"❌ ML Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "status": "failed"}