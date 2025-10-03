# tralis_etl/load/load_dim_customer.py

from sqlalchemy import text, String, Boolean, Integer, DateTime
from Configuration.db_config import get_target_engine
import pandas as pd
from datetime import datetime

def load_dim_customer(df):
    engine = get_target_engine()
    table_name = "DimCustomer"

    print("🔍 Validating DimCustomer data...")
    required_columns = [
        'CustomerID', 'CustomerName', 'CustomerType', 'CustomerCategory',
        'CustomerCountry', 'CustomerCity', 'CustomerRegion',
        'CustomerSegment', 'IsActive', 'LastModifiedOnDate'
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"❌ Missing required column: {col}")

    print("✅ DimCustomer data passed validation.")

    # Format SCD2 fields
    df['EffectiveDate'] = pd.to_datetime(df['LastModifiedOnDate'], errors='coerce').fillna(datetime.today())
    df['ExpiryDate'] = pd.to_datetime('2099-12-31')
    df['IsCurrent'] = True

    df.drop(columns=['LastModifiedOnDate'], inplace=True)

    

    df.to_sql(
        name=table_name,
        con=engine,
        if_exists='append',
        index=False,
        dtype={
            'CustomerID': String(),
            'CustomerName': String(),
            'CustomerType': String(),
            'CustomerCategory': String(),
            'CustomerCountry': Integer(),
            'CustomerCity': String(),
            'CustomerRegion': Integer(),
            'CustomerSegment': String(),
            'IsActive': Boolean(),
            'EffectiveDate': DateTime(),
            'ExpiryDate': DateTime(),
            'IsCurrent': Boolean()
        }
    )

    print(f"✅ Loaded {len(df)} customer records with SCD2 structure after reset.")
