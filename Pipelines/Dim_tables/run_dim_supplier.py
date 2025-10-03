# tralis_etl/Pipelines/run_dim_supplier.py

from extract.Dim_tables.extract_supplier import extract_supplier_data
from transform.Dim_tables.transform_supplier import transform_supplier_data
from load.Dim_tables.load_dim_supplier import load_dim_supplier
from validation.validate_data_quality import validate_supplier_data

def run_pipeline():
    print("🚀 Starting DimSupplier pipeline")
    
    df_raw = extract_supplier_data()
    print(f"📦 Extracted {len(df_raw)} raw supplier records")

    df_clean = transform_supplier_data(df_raw)
    print(f"🧹 Transformed into {len(df_clean)} cleaned supplier records")

    validate_supplier_data(df_clean)
    print("✅ Data validation passed")

    load_dim_supplier(df_clean)
if __name__ == "__main__":
    run_pipeline()