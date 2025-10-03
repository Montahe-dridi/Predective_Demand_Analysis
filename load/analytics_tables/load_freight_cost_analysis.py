import pandas as pd

from Configuration.db_config import get_target_engine

def load_freight_cost(df: pd.DataFrame):
    print("📥 Loading FreightCostAnalysis...")

    engine = get_target_engine()

    try:
        # Write to SQL
        df.to_sql(
            "FreightCostAnalysis",
            engine,
            schema="dbo",        # Adjust if your schema differs
            if_exists="append",  # Use "replace" if you want to drop and recreate each run
            index=False,
            chunksize=5000,
            method="multi"
        )
        print(f"✅ Loaded {len(df)} rows into FreightCostAnalysis.")

    except Exception as e:
        print(f"❌ Error loading FreightCostAnalysis: {e}")
        raise
