# load/load_dim_freight_type.py

from sqlalchemy import String, Boolean, DateTime
from sqlalchemy.sql import text
from Configuration.db_config import get_target_engine
import pandas as pd
from datetime import datetime

def load_dim_freight_type(df):
    print("🚚 Loading data into DimFreightType ...")
    engine = get_target_engine()
    table_name = "DimFreightType"

    
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {table_name}"))
        print("🗑️ Cleared existing data from DimFreightType")

    df_to_load = df[[
        'FreightTypeID', 'FreightTypeName', 'IsActive'
        
    ]]

    df_to_load.to_sql(
        name=table_name,
        con=engine,
        if_exists='append',
        index=False,
        dtype={
            'FreightTypeID': String(),
            'FreightTypeName': String(),
            'IsActive': Boolean(),
            
        }
    )
    print(f"✅ Inserted {len(df)} freight type records .")
