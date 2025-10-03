# tralis_etl/pipelines/run_dim_product_pipeline.py

from extract.Dim_tables.extract_product import extract_product_data
from transform.Dim_tables.transform_product import transform_product_data
from validation.validate_data_quality import validate_dim_product
from load.Dim_tables.load_dim_product import load_dim_product

# logger = setup_logger("dim_product_pipeline")

def run_pipeline():
    print("🚀 Starting DimProduct pipeline")

    # Step 1: Extract
    df_raw = extract_product_data()
    print(f"📦 Extracted {len(df_raw)} raw product records")

    # Step 2: Transform
    df_clean = transform_product_data(df_raw)
    print(f"🧹 Transformed into {len(df_clean)} cleaned product records")

    # Step 3: Validate
    validate_dim_product(df_clean)
    print("✅ Data validation passed")

    # Step 4: Load
    load_dim_product(df_clean)
    print("✅ Loaded data to DimProduct table")

if __name__ == "__main__":
    run_pipeline()
