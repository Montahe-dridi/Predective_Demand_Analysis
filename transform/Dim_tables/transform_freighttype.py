# transform/transform_freight_type.py

import pandas as pd

def transform_freight_type_data(df):
    print("🔄 Transforming freight type data...")
    df = df.copy()
    df['FreightTypeID'] = df['FreightTypeID'].astype(str).str.strip()
    df['FreightTypeName'] = df['FreightTypeName'].fillna('Unknown').str.title()
    df['IsActive'] = True
    df = df.drop_duplicates(subset='FreightTypeID', keep='last')
    print(f"📊 Transformed {len(df)} freight type records")
    return df
