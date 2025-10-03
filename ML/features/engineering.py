# ================================
# ML/features/engineering.py
# ================================

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from typing import List, Tuple


# ================================
# Helper: Month → Season
# ================================
def _map_season(month: int) -> str:
    """Convert month number to season"""
    if pd.isna(month):
        return "Unknown"
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"


# ================================
# Feature Engineering Class
# ================================
class FeatureEngineer:
    """Advanced feature engineering for logistics data"""

    def __init__(self):
        self.encoders = {}
        self.feature_names = {}

    # ================================
    # 🚚 Shipment Features
    # ================================
    def engineer_shipment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Density and efficiency
        df['WeightVolumeDensity'] = df['TotalWeight'] / (df['TotalVolume'] + 1e-6)
        df['ValuePerWeight'] = df['ShipmentValue'] / (df['TotalWeight'] + 1e-6)
        df['ValuePerPackage'] = df['ShipmentValue'] / (df['TotalPackages'] + 1e-6)
        df['PackageDensity'] = df['TotalPackages'] / (df['TotalVolume'] + 1e-6)

        # Dates
        if 'ShipmentDate' in df.columns:
            df['ShipmentDate'] = pd.to_datetime(df['ShipmentDate'], errors='coerce')
            df['Month'] = df['ShipmentDate'].dt.month
            df['Quarter'] = df['ShipmentDate'].dt.quarter
            df['DayOfWeek'] = df['ShipmentDate'].dt.dayofweek
            df['Year'] = df['ShipmentDate'].dt.year
            df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
            df['DayOfMonth'] = df['ShipmentDate'].dt.day
            df['WeekOfYear'] = df['ShipmentDate'].dt.isocalendar().week

            # Season
            df['Season'] = df['Month'].apply(_map_season)

            # Cyclical
            df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
            df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
            df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
            df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

        # Delivery performance
        if all(col in df.columns for col in ['PlannedArrivalDate', 'ActualArrivalDate']):
            df['PlannedArrivalDate'] = pd.to_datetime(df['PlannedArrivalDate'], errors='coerce')
            df['ActualArrivalDate'] = pd.to_datetime(df['ActualArrivalDate'], errors='coerce')
            df['PlannedDuration'] = (df['PlannedArrivalDate'] - df['ShipmentDate']).dt.days
            df['ActualDuration'] = (df['ActualArrivalDate'] - df['ShipmentDate']).dt.days
            df['DeliveryVariance'] = df['ActualDuration'] - df['PlannedDuration']
            df['OnTimeDeliveryFlag'] = (df['DeliveryVariance'] <= 0).astype(int)

        # Categories
        df['ShipmentSize'] = pd.cut(df['TotalWeight'], bins=3, labels=['Small', 'Medium', 'Large'])
        df['ValueCategory'] = pd.cut(df['ShipmentValue'], bins=4, labels=['Low', 'Medium', 'High', 'Premium'])

        return df

    # ================================
    # 💰 Invoice Features
    # ================================
    def engineer_invoice_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Finance
        df['TaxRate'] = df['TaxAmount'] / (df['NetAmount'] + 1e-6)
        df['ProfitMargin'] = (df['NetAmount'] - df['TotalAmount']) / (df['NetAmount'] + 1e-6)

        # Dates
        if 'InvoiceDate' in df.columns:
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
            df['Month'] = df['InvoiceDate'].dt.month
            df['Quarter'] = df['InvoiceDate'].dt.quarter
            df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
            df['Year'] = df['InvoiceDate'].dt.year
            df['IsMonthEnd'] = (df['InvoiceDate'].dt.day > 25).astype(int)
            df['IsQuarterEnd'] = df['InvoiceDate'].dt.month.isin([3, 6, 9, 12]).astype(int)

            # Season
            df['Season'] = df['Month'].apply(_map_season)

            # Cyclical
            df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
            df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

        # Payment
        if all(col in df.columns for col in ['InvoiceDate', 'PaymentDueDate']):
            df['PaymentDueDate'] = pd.to_datetime(df['PaymentDueDate'], errors='coerce')
            df['PaymentPeriod'] = (df['PaymentDueDate'] - df['InvoiceDate']).dt.days
            df['IsOverdue'] = (pd.Timestamp.now() > df['PaymentDueDate']).astype(int)

        # Categories
        df['AmountCategory'] = pd.cut(df['NetAmount'], bins=5, labels=['XS', 'S', 'M', 'L', 'XL'])

        return df

    # ================================
    # 🔍 Feature Selection
    # ================================
    def select_features(self, X: pd.DataFrame, y: pd.Series,
                        task_type: str = 'regression', k: int = 15) -> Tuple[pd.DataFrame, List[str]]:
        X_encoded = X.copy()
        cat_features = X_encoded.select_dtypes(exclude=[np.number]).columns.tolist()

        for col in cat_features:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                X_encoded[col] = self.encoders[col].fit_transform(X_encoded[col].astype(str))
            else:
                X_encoded[col] = self.encoders[col].transform(X_encoded[col].astype(str))

        if task_type == 'regression':
            selector = SelectKBest(score_func=f_regression, k=min(k, X_encoded.shape[1]))
        else:
            selector = SelectKBest(score_func=f_classif, k=min(k, X_encoded.shape[1]))

        X_selected = selector.fit_transform(X_encoded, y)
        selected = X_encoded.columns[selector.get_support()].tolist()

        self.feature_names[task_type] = selected
        return pd.DataFrame(X_selected, columns=selected, index=X.index), selected


# ================================
# Seasonal KPI Generator
# ================================
def generate_seasonal_kpis(shipments: pd.DataFrame, invoices: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    if "ShipmentDate" in shipments.columns:
        shipments["ShipmentDate"] = pd.to_datetime(shipments["ShipmentDate"], errors="coerce")
        shipments["Season"] = shipments["ShipmentDate"].dt.month.apply(_map_season)
        shipment_kpi = shipments.groupby("Season").agg(
            TotalWeight=("TotalWeight", "sum"),
            TotalVolume=("TotalVolume", "sum"),
            Count=("ShipmentDate", "count")
        ).reset_index()
        shipment_kpi.to_csv(os.path.join(output_dir, "shipment_kpis.csv"), index=False)

    if "InvoiceDate" in invoices.columns:
        invoices["InvoiceDate"] = pd.to_datetime(invoices["InvoiceDate"], errors="coerce")
        invoices["Season"] = invoices["InvoiceDate"].dt.month.apply(_map_season)
        invoice_kpi = invoices.groupby("Season").agg(
            TotalAmount=("TotalAmount", "sum"),
            NetAmount=("NetAmount", "sum"),
            Count=("InvoiceDate", "count")
        ).reset_index()
        invoice_kpi.to_csv(os.path.join(output_dir, "invoice_kpis.csv"), index=False)

    if "Season" in shipments.columns and "Season" in invoices.columns:
        combined = pd.merge(
            shipments.groupby("Season").size().reset_index(name="Shipments"),
            invoices.groupby("Season").size().reset_index(name="Invoices"),
            on="Season", how="outer"
        )
        combined.to_csv(os.path.join(output_dir, "combined_kpis.csv"), index=False)
