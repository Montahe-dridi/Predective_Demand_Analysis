# tralis_etl/load/load_dim_equipment.py

from sqlalchemy import text, String, Boolean, Integer, DateTime
from Configuration.db_config import get_target_engine
import pandas as pd

def load_dim_equipment(df):
    engine = get_target_engine()
    table_name = "DimEquipment"

    print("🔍 Validating DimEquipment data...")

    required_columns = [
        'EquipmentID', 'EquipmentType', 'OperationType',
         'IsActive', 'EffectiveDate', 
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"❌ Missing required column: {col}")

    print("✅ DimEquipment data passed validation.")

    
    df['EffectiveDate'] = pd.to_datetime(df['LastModifiedOnDate'], errors='coerce').fillna(pd.to_datetime('1900-01-01'))
    df['ExpiryDate'] = pd.to_datetime("2099-12-31")
    df['IsCurrent'] = True

    df.drop(columns=['LastModifiedOnDate'], inplace=True)

    # Load all current data after reset
    print("📌 Columns in DataFrame:", df.columns.tolist())

    df.to_sql(
        name=table_name,
        con=engine,
        if_exists='append',
        index=False,
        dtype={
            'EquipmentID': String(),
            'EquipmentType': String(),
            'OperationType': String(),
            'IsActive': Boolean(),
            'EffectiveDate': DateTime(),
            'ExpiryDate': DateTime(),
            'IsCurrent': Boolean()
        }
    )

    print(f"✅ Loaded {len(df)} equipment records with SCD2 structure after reset.")
