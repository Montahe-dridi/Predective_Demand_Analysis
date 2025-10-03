# tralis_etl/Pipelines/run_dim_customers.py

from extract.Dim_tables.extract_customer import extract_customer_data
from transform.Dim_tables.transform_customer import transform_customer_data
from load.Dim_tables.load_dim_customer import load_dim_customer
from validation.validate_data_quality import validate_customer_data

def run_pipeline():
    print("🚀 Starting DimCustomer pipeline")
    df_raw = extract_customer_data()
    print(f"📦 Extracted {len(df_raw)} raw customers records")
    df_clean = transform_customer_data(df_raw)
    print(f"🧹 Transformed into {len(df_clean)} cleaned customers records")
    validate_customer_data(df_clean)
    print("✅ Data validation passed")
    load_dim_customer(df_clean)
    print("✅ Loaded data to Dimcustomer table")
    
if __name__ == "__main__":
    run_pipeline()

