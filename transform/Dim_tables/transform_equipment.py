import pandas as pd

def transform_equipment_data(df):
    print("🔄 Transforming equipment data...")
    df = df.copy()

    df['EquipmentID'] = df['EquipmentID'].astype(str).str.strip().str.upper()
    df['EquipmentType'] = df['EquipmentType'].fillna("Unknown").str.strip().str.title()
    df['OperationType'] = df['OperationType'].fillna("Unknown").str.strip().str.title()
    df['IsActive'] = df['IsActive'].fillna(0).astype(bool)
    
# ✅ Drop duplicates 
    df = df.drop_duplicates(subset=['EquipmentID'], keep='last')


    return df[[
    'EquipmentID', 'EquipmentType', 'OperationType',
     'IsActive','LastModifiedOnDate'
    
]]
