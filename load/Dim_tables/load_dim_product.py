# tralis_etl/load/load_dim_product.py

from sqlalchemy import text, String, Boolean, Float, DateTime
from Configuration.db_config import get_target_engine
import pandas as pd
from datetime import datetime

def load_dim_product(df):
    engine = get_target_engine()
    table_name = "DimProduct"

    # Read existing dimension data
    with engine.begin() as conn:
        existing_df = pd.read_sql(f"SELECT * FROM {table_name} WHERE IsCurrent = 1", conn)

    df['EffectiveDate'] = pd.to_datetime(df['LastModifiedOnDate'])
    df['ExpiryDate'] = pd.to_datetime('2099-12-31')  # Changed to avoid overflow error
    df['IsCurrent'] = True

    # Detect changes for SCD Type 2
    merged = df.merge(existing_df, on='ProductID', suffixes=('_new', '_old'), how='left', indicator=True)

    updates = []
    unchanged = []

    for _, row in merged.iterrows():
        if row['_merge'] == 'both':
            if any([
                row['ProductName_new'] != row['ProductName_old'],
                row['ProductCategory_new'] != row['ProductCategory_old'],
                row['ProductType_new'] != row['ProductType_old'],
                row['IsStockable_new'] != row['IsStockable_old'],
                row['IsActiveForSale_new'] != row['IsActiveForSale_old'],
                row['IsActiveForPurchase_new'] != row['IsActiveForPurchase_old'],
                row['IsActive_new'] != row['IsActive_old']
            ]):
                updates.append(row)
            else:
                unchanged.append(row)
        elif row['_merge'] == 'left_only':
            updates.append(row)

    to_insert = pd.DataFrame([{
    'ProductID': r['ProductID'],
    'ProductName': r['ProductName_new'],
    'ProductCategory': r['ProductCategory_new'],
    'ProductType': r['ProductType_new'],
    'IsStockable': r['IsStockable_new'],
    'IsActiveForSale': r['IsActiveForSale_new'],
    'IsActiveForPurchase': r['IsActiveForPurchase_new'],
    'IsActive': r['IsActive_new'],
    'EffectiveDate': pd.to_datetime(r['LastModifiedOnDate']),

    'ExpiryDate': pd.to_datetime('2099-12-31'),
    'IsCurrent': True
} for r in updates])


    # Expire old records
    with engine.begin() as conn:
        for r in updates:
            conn.execute(
                text(f"""
                    UPDATE {table_name}
                    SET ExpiryDate = :expiry, IsCurrent = 0
                    WHERE ProductID = :pid AND IsCurrent = 1
                """),
                {"expiry": datetime.now(), "pid": r['ProductID']}
            )

    # Load new records
    if not to_insert.empty:
        to_insert.to_sql(
            name=table_name,
            con=engine,
            if_exists='append',
            index=False,
            dtype={
                'ProductID': String(),
                'ProductName': String(),
                'ProductCategory': String(),
                'ProductType': String(),
                'IsStockable': Boolean(),
                'IsActiveForSale': Boolean(),
                'IsActiveForPurchase': Boolean(),
                'IsActive': Boolean(),
                'EffectiveDate': DateTime(),
                'ExpiryDate': DateTime(),
                'IsCurrent': Boolean()
            }
        )
        print(f"✅ Inserted {len(to_insert)} new product records with SCD Type 2 tracking.")
    else:
        print("ℹ️ No new or changed product records to load.")
