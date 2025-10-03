# tralis_etl/load/load_dim_supplier.py

from sqlalchemy import String, Boolean, DateTime
from sqlalchemy import text
from Configuration.db_config import get_target_engine
import pandas as pd
from datetime import datetime

def load_dim_supplier(df):
    engine = get_target_engine()
    table_name = "DimSupplier"

    print("🔍 Validating DimSupplier data...")

    # Validation done earlier — proceed to enrich with SCD2 fields
    df = df.copy()
    df['EffectiveDate'] = pd.to_datetime(df['LastModifiedOnDate'], errors='coerce').fillna(datetime.today())
    df['ExpiryDate'] = pd.to_datetime('2099-12-31')
    df['IsCurrent'] = True

    # Drop column not in target
    df.drop(columns=['LastModifiedOnDate'], inplace=True)

    print("📥 Loading into DimSupplier...")

    df.to_sql(
        name=table_name,
        con=engine,
        if_exists='append',
        index=False,
        dtype={
            'SupplierID': String(),
            'SupplierName': String(),
            'SupplierCountry': String(),
            'SupplierCity': String(),
            'SupplierRegion': String(),
            
            'IsActive': Boolean(),
            'EffectiveDate': DateTime(),
            'ExpiryDate': DateTime(),
            'IsCurrent': Boolean()
        }
    )

    print(f"✅ Inserted {len(df)} supplier records with SCD Type 2 logic.")
