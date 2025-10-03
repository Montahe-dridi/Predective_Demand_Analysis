# tralis_etl/extract/extract_equipment.py

import pandas as pd
from sqlalchemy import create_engine
from Configuration.db_config import get_source_engine

def extract_equipment_data():
    print("📥 Extracting supplier data from source...")
    engine = get_source_engine()
    query = """
    SELECT
    CAST(E.ID AS VARCHAR) AS EquipmentID,
    E.Designation AS EquipmentType,
    T.Designation AS OperationType,
    E.IsActive,
    E.LastModifiedOnDate
FROM dbo.TRADE_Equipement E
LEFT JOIN dbo.TRADE_TypeOperationCotation T
    ON E.ID_typeOp = T.ID
    """

    df = pd.read_sql(query, engine)
    print(f"✅ Extracted {len(df)} equipment records")
    return df
