# ==========================================
# ML/utils/seasonal_kpi_exporter.py
# ==========================================

import os
import pandas as pd
import numpy as np
from ML.config.model_config import ModelConfig

class SeasonalKPIExporter:
    """Generate and export seasonal KPIs and detailed CSVs for Power BI"""

    def __init__(self):
        self.config = ModelConfig()
        self.output_dir = os.path.join(self.config.OUTPUT_DIR, "seasonal_kpis")
        self.detailed_dir = os.path.join(self.config.OUTPUT_DIR, "detailed_exports")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.detailed_dir, exist_ok=True)

    def export(self, shipments: pd.DataFrame, invoices: pd.DataFrame) -> dict:
        """Create seasonal KPIs and detailed CSVs for Power BI"""
        results = {}

        # Shipments KPI
        if not shipments.empty and "Season" in shipments.columns:
            shipments_kpi = (
                shipments.groupby("Season")
                .agg(
                    TotalShipments=("ShipmentID", "count"),
                    AvgWeight=("TotalWeight", "mean"),
                    AvgValue=("ShipmentValue", "mean"),
                )
                .reset_index()
            )
            shipments_file = os.path.join(self.output_dir, "shipments_by_season.csv")
            shipments_kpi.to_csv(shipments_file, index=False)
            results["shipments_by_season"] = shipments_file

        # Invoices KPI
        if not invoices.empty and "Season" in invoices.columns:
            invoices_kpi = (
                invoices.groupby("Season")
                .agg(
                    TotalInvoices=("InvoiceID", "count"),
                    TotalNetAmount=("NetAmount", "sum"),
                    TotalTax=("TaxAmount", "sum"),
                )
                .reset_index()
            )
            invoices_file = os.path.join(self.output_dir, "invoices_by_season.csv")
            invoices_kpi.to_csv(invoices_file, index=False)
            results["invoices_by_season"] = invoices_file

        print(f"✅ Seasonal KPIs exported: {results}")
        return results
