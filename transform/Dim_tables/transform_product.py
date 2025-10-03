# tralis_etl/transform/transform_product.py

import pandas as pd

def transform_product_data(df):
    df = df.copy()

    # Clean and standardize string columns
    df['ProductID'] = df['ProductID'].astype(str).str.strip().str.upper()
    df['ProductName'] = df['ProductName'].fillna('Unknown').str.title()
    df['ProductCategory'] = df['ProductCategory'].fillna('Unknown')
    df['ProductType'] = df['ProductType'].fillna('Goods')

    # Convert flag columns to boolean
    df['IsStockable'] = df['IsStockable'].astype(bool)
    df['IsActiveForSale'] = df['IsActiveForSale'].astype(bool)
    df['IsActiveForPurchase'] = df['IsActiveForPurchase'].astype(bool)
    df['IsActive'] = df['IsActive'].astype(bool)

    # Drop duplicates by ProductID, keeping the latest modified version if needed
    df = df.drop_duplicates(subset=['ProductID'], keep='last')

    # ✅ Include LastModifiedOnDate for SCD tracking
    return df[[
        'ProductID', 'ProductName', 'ProductCategory', 'ProductType',
        'IsStockable', 'IsActiveForSale', 'IsActiveForPurchase', 'IsActive',
        'LastModifiedOnDate'  
    ]]
