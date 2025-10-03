import pandas as pd
from transform.analytics_tables.transform_freight_cost_analysis import transform_freight_cost
from load.analytics_tables.load_freight_cost_analysis import load_freight_cost
from validation.validate_data_quality import validate_freight_cost
from Configuration.db_config import get_target_engine


def run_freight_cost_pipeline():
    engine = get_target_engine()

    # Extract: pull from FactShipments
    fact_shipments = pd.read_sql(
        """
        SELECT 
            ShipmentKey, 
            FreightCost, 
            Weight, 
            Distance, 
            CarrierKey
        FROM FactShipments
        """,
        engine
    )

    # Transform
    df_transformed = transform_freight_cost(fact_shipments)

    # Validate
    valid, issues = validate_freight_cost(df_transformed)
    if not valid:
        print("❌ Validation failed. Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        print("⚠️ FreightCostAnalysis load skipped due to validation errors.")
        return

    # Load
    load_freight_cost(df_transformed)
