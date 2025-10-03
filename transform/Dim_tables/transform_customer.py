# tralis_etl/transform/transform_customer.py

import pandas as pd

def transform_customer_data(df):
    print("🔄 Transforming customer data...")
    df = df.copy()

    # Clean and standardize string columns
    df['CustomerID'] = df['CodeClient'].astype(str).str.strip().str.upper()
    df['CustomerName'] = df['RaisonSociale'].fillna('Unknown').str.strip().str.title()
    df['CustomerCity'] = df['Ville'].fillna('').str.strip()

    # Classify type and category
    df['CustomerType'] = df['NatureSociete'].apply(lambda x: 'Independant' if x == 0 else 'Entreprise')
    df['CustomerCategory'] = df['NatureSociete'].apply(lambda x: 'B2C' if x == 0 else 'B2B')

    # Convert numeric and boolean flags
    df['CustomerCountry'] = df['ID_Pays'].fillna(0).astype(int)
    df['CustomerRegion'] = df['ID_Departement'].fillna(0).astype(int)
    df['CustomerSegment'] = df['IsClientRegulier'].apply(lambda x: 'Regular' if x else 'Occasional')
    df['IsActive'] = df['Etat'].apply(lambda x: True if x == 1 else False)

    # ✅ Drop duplicates (keep last by CustomerID)
    df = df.drop_duplicates(subset=['CustomerID'], keep='last')

    # ✅ Return required clean columns with LastModifiedOnDate for SCD
    return df[[
        'CustomerID', 'CustomerName', 'CustomerType', 'CustomerCategory',
        'CustomerCountry', 'CustomerCity', 'CustomerRegion',
        'CustomerSegment', 'IsActive', 'LastModifiedOnDate'
    ]]
