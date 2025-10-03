# extract/extract_freight_type.py

from sqlalchemy import create_engine
import pandas as pd
from Configuration.db_config import get_source_engine

def extract_freight_type_data():
    print("📥 Extracting freight type data from source...")
    engine = get_source_engine()
    query = """
    SELECT 
        CAST(ID AS VARCHAR) AS FreightTypeID,
        Designation AS FreightTypeName
    FROM dbo.Exploitation_Freight
    """
    df = pd.read_sql(query, engine)
    print(f"✅ Extracted {len(df)} freight type records")
    return df
