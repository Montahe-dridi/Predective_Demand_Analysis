from extract.Dim_tables.extract_equipment import extract_equipment_data
from transform.Dim_tables.transform_equipment import transform_equipment_data
from load.Dim_tables.load_dim_equipment import load_dim_equipment
from validation.validate_data_quality import validate_equipment_data
import pandas as pd

def run_pipeline():
    print("🚀 Starting DimEquipment pipeline")

    # 1. Extract
    df_raw = extract_equipment_data()

    # 2. Transform
    df_clean = transform_equipment_data(df_raw)

    # ✅ Add EffectiveDate with fallback for nulls
    if 'LastModifiedOnDate' in df_clean.columns:
        df_clean['EffectiveDate'] = pd.to_datetime(df_clean['LastModifiedOnDate'], errors='coerce')
        df_clean['EffectiveDate'].fillna(pd.to_datetime('1900-01-01'), inplace=True)  # Fallback if null
    else:
        df_clean['EffectiveDate'] = pd.to_datetime('1900-01-01')  # Default for all

    # 3. Validate
    validate_equipment_data(df_clean)

    # 4. Load
    load_dim_equipment(df_clean)
