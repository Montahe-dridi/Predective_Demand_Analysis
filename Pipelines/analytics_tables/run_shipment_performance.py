import pandas as pd
from transform.analytics_tables.transform_shipment_performance import transform_shipment_performance
from load.analytics_tables.load_shipment_performance import load_shipment_performance
from validation.validate_data_quality import validate_shipment_performance_data
from Configuration.db_config import get_target_engine

def run_shipment_performance_pipeline():
    print("Starting ShipmentPerformanceMetrics pipeline")
    engine = get_target_engine()

    # FIXED: Extract using ShipmentKey instead of ShipmentID
    fact_shipments = pd.read_sql("""
        SELECT ShipmentKey, PlannedArrivalDate, ActualArrivalDate 
        FROM FactShipments
        WHERE ActualArrivalDate IS NOT NULL
    """, engine)
    
    print(f"Extracted {len(fact_shipments)} rows from FactShipments")
    print("Source data null counts:")
    print(fact_shipments.isnull().sum())

    # Transform - using the filter nulls approach
    df_transformed = transform_shipment_performance(fact_shipments)
    
    print(f"Transformed data shape: {df_transformed.shape}")

    # Validate
    validation_passed = validate_shipment_performance_data(df_transformed)

    # Load only if validation passed
    if validation_passed:
        load_shipment_performance(df_transformed)
        print("ShipmentPerformanceMetrics load complete.")
    else:
        print("ShipmentPerformanceMetrics load skipped due to validation issues.")