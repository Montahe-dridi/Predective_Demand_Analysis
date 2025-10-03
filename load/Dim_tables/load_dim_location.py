# tralis_etl/load/load_dim_location.py

from sqlalchemy import String, Boolean, DateTime
from Configuration.db_config import get_target_engine
import pandas as pd
from datetime import datetime

def load_dim_location(df):
    engine = get_target_engine()
    table_name = "DimLocation"

    df = df.copy()
    df['EffectiveDate'] = pd.to_datetime(df['CreatedOnDate'], errors='coerce')

# Replace NULLs with today's date and log warning
    missing_dates = df['EffectiveDate'].isnull().sum()
    if missing_dates > 0:
     print(f"⚠️ {missing_dates} records missing CreatedOnDate. Assigned  neutral date as fallback.")
    
# Fallback to 1900-01-01 for nulls (so it's easy to detect/filter later)
    df['EffectiveDate'].fillna(pd.to_datetime('1900-01-01'), inplace=True)
    df['ExpiryDate'] = pd.to_datetime('2099-12-31')
    df['IsCurrent'] = True

    df.drop(columns=['CreatedOnDate'], inplace=True)

    df.to_sql(
        name=table_name,
        con=engine,
        if_exists='append',
        index=False,
        dtype={
            'LocationID': String(),
            'LocationName': String(),
            'LocationType': String(),
            'IsActive': Boolean(),
            'EffectiveDate': DateTime(),
            'ExpiryDate': DateTime(),
            'IsCurrent': Boolean()
        }
    )

    print(f"✅ Inserted {len(df)} location records with SCD Type 2 tracking.")
