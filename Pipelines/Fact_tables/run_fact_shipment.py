from extract.Fact_tables.extract_fact_shipment import (
    extract_fact_shipment,
    extract_dim_customer_keys,
    extract_dim_date_keys,         
    extract_dim_equipment_keys,
    extract_dim_freight_keys,
    extract_dim_location_keys
)

from transform.Fact_tables.transform_fact_shipment import transform_fact_shipment
from load.Fact_tables.load_fact_shipment import load_fact_shipment
from validation.validate_data_quality import validate_fact_shipment


def run_pipeline():
    print("🚛 Starting FactShipment pipeline")

    # Step 1: Extract
    df_raw = extract_fact_shipment()

    # ✅ Extract real DimDate from SQL (not generated)
    dim_dates = extract_dim_date_keys()

    dim_customers = extract_dim_customer_keys()
    print("✅ Customer columns:", dim_customers.columns.tolist())
    
    dim_equipment = extract_dim_equipment_keys()
    print("✅ equipment columns:", dim_equipment.columns.tolist())
    
    dim_freight = extract_dim_freight_keys()
    print("✅ freight columns:", dim_freight.columns.tolist())
    
    dim_locations = extract_dim_location_keys()
    print("✅ location columns:", dim_locations.columns.tolist())

    print(f"✅ Extracted {len(df_raw)} raw shipment records")

    # Step 2: Transform
    df_transformed = transform_fact_shipment(
        df_raw, dim_dates, dim_customers, 
        dim_equipment, dim_freight, dim_locations
    )

    # Step 3: Validate
    validate_fact_shipment(df_transformed)

    # Step 4: Load
    load_fact_shipment(df_transformed)

    print("✅ FactShipment pipeline completed.")
