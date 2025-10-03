# tralis_etl/Pipelines/run_dim_location.py

from extract.Dim_tables.extract_location import extract_location_data
from transform.Dim_tables.transform_location import transform_location_data
from validation.validate_data_quality import validate_location_data
from load.Dim_tables.load_dim_location import load_dim_location

def run_pipeline():
    print("🚀 Starting DimLocation pipeline")
    df_raw = extract_location_data()
    print(f"📦 Extracted {len(df_raw)} raw location records")

    df_clean = transform_location_data(df_raw)
    print(f"🧹 Transformed into {len(df_clean)} cleaned location records")

    validate_location_data(df_clean)
    print("✅ Data validation passed")

    load_dim_location(df_clean)
