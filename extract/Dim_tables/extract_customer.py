# tralis_etl/extract/extract_customer.py

import pandas as pd
from sqlalchemy import create_engine
from Configuration.db_config import get_source_engine

def extract_customer_data():
    print("📥 Extracting customer data from source...")
    engine = get_source_engine()
    
    query = """
        SELECT 
            ID, CodeClient, RaisonSociale, NatureSociete, 
            ID_Pays, Ville, ID_Departement,
            IsClientRegulier, Etat,
            CreatedOnDate, LastModifiedOnDate
        FROM dbo.Tiers_Tiers
        WHERE IsClient = 1
    """
    
    df = pd.read_sql(query, engine)
    print(f"✅ Extracted {len(df)} customer records")
    return df
