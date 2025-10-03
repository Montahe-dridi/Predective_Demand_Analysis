def extract_payment_term_data():
    from sqlalchemy import create_engine
    from Configuration.db_config import get_source_engine

    import pandas as pd

    print("📥 Extracting payment term data from source...")
    engine = get_source_engine()
    query = """
    SELECT 
        CAST(ID AS VARCHAR) AS PaymentTermID,
        Designation AS PaymentTermName,
        1 AS IsActive,
        NumOfDays AS PaymentDays 
    FROM dbo.Purchasing_PaymentTerms 
    WHERE ID IS NOT NULL
    """
    df = pd.read_sql(query, engine)
    print(f"✅ Extracted {len(df)} payment term records")
    return df
