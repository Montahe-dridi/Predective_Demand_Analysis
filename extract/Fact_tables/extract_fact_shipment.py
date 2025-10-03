import pandas as pd
from sqlalchemy import create_engine
from Configuration.db_config import get_source_engine, get_target_engine

# ✅ 1. Extract DimDate directly from target SQL Server (avoid using generated calendar)
def extract_dim_date_keys():
    engine = get_target_engine()
    query = "SELECT DateKey FROM DimDate"
    df = pd.read_sql(query, engine)
    df['DateKey'] = df['DateKey'].astype(int)
    print(f"📅 Extracted {len(df)} date records from DimDate")
    return df

# ✅ 2. Extract FactShipment source data
def extract_fact_shipment():
    print("📥 Extracting shipment data ...")
    engine = get_source_engine()
    query = """
    SELECT
        bi.ID AS ShipmentID,
        ISNULL(dos.DateDepart, GETDATE()) AS ShipmentDate,
        dos.DateArrivee AS PlannedArrivalDate,
        dos.DateDepartReel AS ActualDepartureDate,
        dos.DateArriveeReel AS ActualArrivalDate,

        bi.Poids AS TotalWeight,
        bi.Volume AS TotalVolume,
        bi.nbColis AS TotalPackages,
        dos.ValMarchandise AS ShipmentValue,

        CAST(COALESCE(bi.Client, '') AS NVARCHAR) AS CustomerName,
        CAST(dos.ID_LieuChargement AS NVARCHAR) AS OriginLocationID,
        CAST(dos.ID_LieuLivraison AS NVARCHAR) AS DestinationLocationID,
        CAST(bi.ID_Equipement AS NVARCHAR) AS EquipmentID,
        CAST(dos.ID_Freight AS NVARCHAR) AS FreightTypeID,

        CASE WHEN dos.DateArriveeReel IS NOT NULL THEN 1 ELSE 0 END AS IsDelivered,
        DATEDIFF(DAY, dos.DateDepart, dos.DateArriveeReel) AS DurationDays

    FROM dbo.Exploitation_Expedition_BI bi
    LEFT JOIN dbo.Exploitation_vwDossier dos ON bi.ID_Dossier = dos.ID
    WHERE bi.ID IS NOT NULL
    """
    
    df = pd.read_sql(query, engine)
    print(f"✅ Extracted {len(df)} shipment records")
    return df

# ✅ 3. Customer dimension
def extract_dim_customer_keys():
    engine = get_target_engine()
    query = "SELECT CustomerKey, CustomerName FROM DimCustomer WHERE IsCurrent = 1"
    return pd.read_sql(query, engine)

# ✅ 4. Equipment dimension
def extract_dim_equipment_keys():
    engine = get_target_engine()
    query = "SELECT EquipmentKey, EquipmentID FROM DimEquipment WHERE IsCurrent = 1"
    return pd.read_sql(query, engine)

# ✅ 5. Freight dimension
def extract_dim_freight_keys():
    engine = get_target_engine()
    query = "SELECT FreightTypeKey, FreightTypeID FROM DimFreightType"
    return pd.read_sql(query, engine)

# ✅ 6. Location dimension
def extract_dim_location_keys():
    engine = get_target_engine()
    query = "SELECT LocationKey, LocationID FROM DimLocation WHERE IsCurrent = 1"
    return pd.read_sql(query, engine)
