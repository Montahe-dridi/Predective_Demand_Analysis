import pandas as pd
from sqlalchemy import create_engine
from Configuration.db_config import get_source_engine

def extract_location_data():
    print("📥 Extracting supplier data from source...")
    engine = get_source_engine()
    
    union_query = """
    SELECT DISTINCT
        CAST(ID AS VARCHAR) AS LocationID,
        Libelle AS LocationName,
        'Port' AS LocationType,
        CreatedOnDate,
        
        1 AS IsActive
    FROM dbo.Framework_Port

    UNION

    SELECT DISTINCT
        CAST(ID AS VARCHAR) AS LocationID,
        Libelle AS LocationName,
        'Airport' AS LocationType,
        CreatedOnDate,
        1 AS IsActive
    FROM dbo.Framework_Aeroport

    UNION

    SELECT DISTINCT
        CAST(ID AS VARCHAR) AS LocationID,
        Libelle AS LocationName,
        'City' AS LocationType,
        CreatedOnDate,
        
        
        1 AS IsActive
    FROM dbo.Framework_Ville
    """
    df = pd.read_sql(union_query, engine)
    print(f"✅ Extracted {len(df)} location records")
    return df
