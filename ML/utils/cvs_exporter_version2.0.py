# ==========================================
# ML/utils/csv_based_exporter.py - Final ML-integrated version
# ==========================================

import os
import pandas as pd
import numpy as np
import warnings
import joblib
from datetime import datetime

warnings.filterwarnings('ignore')

# Import your ML foundation
from ML.data.loaders import DataLoader
from ML.data.preprocessors import DataPreprocessor
from ML.features.engineering import FeatureEngineer
from ML.evaluation.metrics import ModelEvaluator


class CSVBasedExporter:
    """Generate BI-ready CSVs linked to trained ML models
    
    - Loads real data from DataLoader
    - Preprocesses & engineers features
    - Uses trained joblib models for predictions
    - Exports CSVs for Power BI dashboards
    """

    def __init__(self):
        self.output_dir = os.path.join("ML", "outputs", "corrected_exports_final")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📁 Exporting ML-integrated files to: {self.output_dir}")

        # Initialize ML components
        self.loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.engineer = FeatureEngineer()
        self.evaluator = ModelEvaluator()

        # Map models to tasks
        self.models = {
            "delivery_variance": "ML/saved_models/gradient_boost_regression.joblib",   # best from regression
            "invoice_profit": "ML/saved_models/lasso_regression.joblib",              # best from profit regression
            "on_time_delivery": "ML/saved_models/gradient_boost_classification.joblib", # best from classification
            "payment_status": "ML/saved_models/random_forest_classification.joblib"   # best for PaymentStatus
        }

    # ================================
    # Load and prepare base data
    # ================================
    def load_data(self):
        print("📊 Loading and preprocessing data...")

        shipments = self.loader.load_shipments()
        invoices = self.loader.load_invoices()

        shipments = self.preprocessor.preprocess_shipments(shipments)
        invoices = self.preprocessor.preprocess_invoices(invoices)

        shipments = self.engineer.engineer_shipment_features(shipments)
        invoices = self.engineer.engineer_invoice_features(invoices)

        return shipments, invoices

    # ================================
    # Export files
    # ================================
    def export_corrected_files(self):
        shipments, invoices = self.load_data()
        results = {}

        # ================================
        # 1. Executive Summary
        # ================================
        total_customers = invoices["CustomerKey"].nunique() if "CustomerKey" in invoices else 0
        total_transactions = len(invoices)
        avg_transaction_value = invoices["NetAmount"].mean() if "NetAmount" in invoices else 0
        avg_profit_margin = invoices["ProfitMargin"].mean() if "ProfitMargin" in invoices else 0

        exec_summary = pd.DataFrame({
            "Metric": [
                "Customer Entities Served",
                "Total Transactions Processed", 
                "Total Shipments Managed",
                "Average Transaction Value (TND)",
                "Average Profit Margin (%)",
            ],
            "Value": [
                total_customers,
                total_transactions,
                len(shipments),
                round(avg_transaction_value, 2),
                round(avg_profit_margin, 2),
            ],
        })
        file1 = os.path.join(self.output_dir, "executive_summary.csv")
        exec_summary.to_csv(file1, index=False)
        results["executive_summary"] = file1

        # ================================
        # 2. Customer Performance (base KPIs)
        # ================================
        customer_perf = invoices.groupby("CustomerKey").agg(
            TotalInvoices=("InvoiceDate", "count"),
            TotalRevenue=("NetAmount", "sum"),
            AvgInvoiceValue=("NetAmount", "mean"),
            TotalProfit=("InvoiceProfit", "sum"),
            ProfitMargin=("ProfitMargin", "mean"),
        ).reset_index()

        file2 = os.path.join(self.output_dir, "customer_performance.csv")
        customer_perf.to_csv(file2, index=False)
        results["customer_performance"] = file2

        # ================================
        # 3. Delivery Variance Prediction
        # ================================
        try:
            model = joblib.load(self.models["delivery_variance"])
            features, _ = self.engineer.select_features(
                shipments.drop(columns=["DeliveryVariance"], errors="ignore"),
                shipments.get("DeliveryVariance"),
                task_type="regression"
            )
            shipments["PredictedDeliveryVariance"] = model.predict(features)
        except Exception as e:
            print(f"⚠️ Delivery variance model not available: {e}")
            shipments["PredictedDeliveryVariance"] = np.nan

        file3 = os.path.join(self.output_dir, "delivery_variance_predictions.csv")
        shipments.to_csv(file3, index=False)
        results["delivery_variance_predictions"] = file3

        # ================================
        # 4. Invoice Profit Prediction
        # ================================
        try:
            model = joblib.load(self.models["invoice_profit"])
            features, _ = self.engineer.select_features(
                invoices.drop(columns=["InvoiceProfit"], errors="ignore"),
                invoices.get("InvoiceProfit"),
                task_type="regression"
            )
            invoices["PredictedProfit"] = model.predict(features)
        except Exception as e:
            print(f"⚠️ Invoice profit model not available: {e}")
            invoices["PredictedProfit"] = np.nan

        file4 = os.path.join(self.output_dir, "invoice_profit_predictions.csv")
        invoices.to_csv(file4, index=False)
        results["invoice_profit_predictions"] = file4

        # ================================
        # 5. On-Time Delivery Classification
        # ================================
        try:
            model = joblib.load(self.models["on_time_delivery"])
            features, _ = self.engineer.select_features(
                shipments.drop(columns=["OnTimeDeliveryFlag"], errors="ignore"),
                shipments.get("OnTimeDeliveryFlag"),
                task_type="classification"
            )
            shipments["OnTimePrediction"] = model.predict(features)
        except Exception as e:
            print(f"⚠️ On-time delivery model not available: {e}")
            shipments["OnTimePrediction"] = "Unknown"

        file5 = os.path.join(self.output_dir, "on_time_delivery_predictions.csv")
        shipments.to_csv(file5, index=False)
        results["on_time_delivery_predictions"] = file5

        # ================================
        # 6. Payment Status Classification
        # ================================
        try:
            model = joblib.load(self.models["payment_status"])
            features, _ = self.engineer.select_features(
                invoices.drop(columns=["PaymentStatus"], errors="ignore"),
                invoices.get("PaymentStatus"),
                task_type="classification"
            )
            invoices["PredictedPaymentStatus"] = model.predict(features)
        except Exception as e:
            print(f"⚠️ Payment status model not available: {e}")
            invoices["PredictedPaymentStatus"] = "Unknown"

        file6 = os.path.join(self.output_dir, "payment_status_predictions.csv")
        invoices.to_csv(file6, index=False)
        results["payment_status_predictions"] = file6

        # ================================
        # 7. ML Model Performance (evaluations)
        # ================================
        ml_performance = []

        # Evaluate profit model if ground truth exists
        if "InvoiceProfit" in invoices and "PredictedProfit" in invoices:
            eval_results = self.evaluator.evaluate_regression(
                invoices["InvoiceProfit"], invoices["PredictedProfit"], model_name="Invoice Profit"
            )
            ml_performance.append(eval_results)

        # Evaluate delivery variance model if ground truth exists
        if "DeliveryVariance" in shipments and "PredictedDeliveryVariance" in shipments:
            eval_results = self.evaluator.evaluate_regression(
                shipments["DeliveryVariance"], shipments["PredictedDeliveryVariance"], model_name="Delivery Variance"
            )
            ml_performance.append(eval_results)

        # Evaluate classification models
        if "OnTimeDeliveryFlag" in shipments and "OnTimePrediction" in shipments:
            eval_results = self.evaluator.evaluate_classification(
                shipments["OnTimeDeliveryFlag"], shipments["OnTimePrediction"], model_name="On-Time Delivery"
            )
            ml_performance.append(eval_results)

        if "PaymentStatus" in invoices and "PredictedPaymentStatus" in invoices:
            eval_results = self.evaluator.evaluate_classification(
                invoices["PaymentStatus"], invoices["PredictedPaymentStatus"], model_name="Payment Status"
            )
            ml_performance.append(eval_results)

        ml_perf_df = pd.DataFrame(ml_performance)
        file7 = os.path.join(self.output_dir, "ml_model_performance.csv")
        ml_perf_df.to_csv(file7, index=False)
        results["ml_model_performance"] = file7

        print(f"\n✅ Generated {len(results)} ML-integrated CSV files for Power BI:")
        for name in results.keys():
            print(f"  - {name}.csv")

        return results


if __name__ == "__main__":
    exporter = CSVBasedExporter()
    exporter.export_corrected_files()
