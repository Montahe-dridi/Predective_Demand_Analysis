# tralis_etl/transform/transform_supplier_data.py

import pandas as pd

def transform_supplier_data(df):
    print("🔄 Transforming supplier data...")
    df = df.copy()

    df['SupplierID'] = df['CodeSupplier'].astype(str).str.strip().str.upper()
    df['SupplierName'] = df['RaisonSociale'].fillna('Unknown').str.strip().str.title()
    df['SupplierType'] = df['NatureSociete'].apply(lambda x: 'Independant' if x == 0 else 'Entreprise')
    
    df['SupplierCountry'] = df['ID_Pays'].fillna(0).astype(int)
    df['SupplierCity'] = df['Ville'].fillna('').str.strip()
    df['SupplierRegion'] = df['ID_Departement'].fillna(0).astype(int)
    
    df['IsActive'] = df['Etat'].apply(lambda x: True if x == 1 else False)

    # Drop missing IDs
    df = df.dropna(subset=['SupplierID'])

    return df[[
        'SupplierID', 'SupplierName', 'SupplierType', 
        'SupplierCountry', 'SupplierCity', 'SupplierRegion',
         'IsActive', 'LastModifiedOnDate'
    ]]
