# tralis_etl/extract/extract_supplier.py

import pandas as pd
from sqlalchemy import create_engine
from Configuration.db_config import get_source_engine


def extract_supplier_data():
    print("📥 Extracting supplier data from source...")
    engine = get_source_engine()

    query = """
        SELECT 
    ID, 
    CodeSupplier, 
    RaisonSociale, 
    NatureSociete, 
    ID_Pays, 
    Ville, 
    ID_Departement, 
    Etat, 
    CreatedOnDate, 
    LastModifiedOnDate
FROM dbo.Tiers_Tiers
WHERE IsSupplier = 1

    """

    df = pd.read_sql(query, engine)
    print(f"✅ Extracted {len(df)} supplier records")
    return df
