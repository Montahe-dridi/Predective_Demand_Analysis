# tralis_etl/extract/extract_product.py

import pandas as pd
from sqlalchemy.sql import text
from Configuration.db_config import get_source_engine

def extract_product_data():
    engine = get_source_engine()
    query = text("""
        SELECT
            CAST(Code AS NVARCHAR(50)) AS ProductID,
            TRIM(ISNULL(Nom, '')) AS ProductName,
            CAST(TypeCode AS NVARCHAR(50)) AS ProductCategory,
            CASE 
                WHEN ISNULL(IsServices, 0) = 1 THEN 'Service'
                ELSE 'Goods'
            END AS ProductType,
            ISNULL(Stockable, 0) AS IsStockable,
            ISNULL(EtatVente, 0) AS IsActiveForSale,
            ISNULL(EtatAchat, 0) AS IsActiveForPurchase,
            ISNULL(IsFret, 0) AS IsActive,
            LastModifiedOnDate
        FROM dbo.Materials_Materials
        WHERE Code IS NOT NULL AND LTRIM(RTRIM(Code)) <> ''
    """)
    df = pd.read_sql(query, engine)
    return df
