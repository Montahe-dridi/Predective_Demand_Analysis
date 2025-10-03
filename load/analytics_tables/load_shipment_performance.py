
from sqlalchemy import text, Integer, Float, DateTime, String, Boolean, DECIMAL
from Configuration.db_config import get_target_engine
import pandas as pd



def load_shipment_performance(df: pd.DataFrame):
    engine = get_target_engine()
    print(f"📥 Loading {len(df)} rows into ShipmentPerformanceMetrics...")
    df.to_sql("ShipmentPerformanceMetrics", engine, if_exists="append", index=False)
    print("✅ ShipmentPerformanceMetrics load complete.")