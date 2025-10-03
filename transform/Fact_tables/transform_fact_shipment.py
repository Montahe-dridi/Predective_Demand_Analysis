import pandas as pd

def transform_fact_shipment(df, dim_dates, dim_customers,
                             dim_equipment, dim_freight, dim_locations):
    print("🔄 Transforming FactShipment data...")
    df = df.copy()

    #  1. Convert date fields
    date_cols = ['ShipmentDate', 'PlannedArrivalDate', 'ActualDepartureDate', 'ActualArrivalDate']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # 📌 2. Create DateKey (yyyymmdd) and validate against DimDate
    df['DateKey'] = df['ShipmentDate'].dt.strftime('%Y%m%d').astype(int)
    dim_dates['DateKey'] = dim_dates['DateKey'].astype(int)
    valid_datekeys = set(dim_dates['DateKey'])

    invalid_dates = df[~df['DateKey'].isin(valid_datekeys)]
    if not invalid_dates.empty:
        invalid_dates[['ShipmentID', 'ShipmentDate', 'DateKey']].to_csv("invalid_datekeys.csv", index=False)
        print(f"🚨 Found {len(invalid_dates)} rows with invalid DateKeys. Saved to 'invalid_datekeys.csv'")

    df = df[df['DateKey'].isin(valid_datekeys)]
    print(f"✅ Kept {len(df)} rows with valid DateKeys")

    # 👤 3. Join with DimCustomer
    df['CustomerName'] = df['CustomerName'].str.lower().str.strip()
    dim_customers['CustomerName'] = dim_customers['CustomerName'].str.lower().str.strip()

    df = df.merge(dim_customers[['CustomerKey', 'CustomerName']], on='CustomerName', how='left')

    unmatched_customers = df[df['CustomerKey'].isna()][['CustomerName']].drop_duplicates()
    if not unmatched_customers.empty:
        unmatched_customers.to_csv("unmatched_customers.csv", index=False)
        print(f"❌ {len(unmatched_customers)} unmatched customers written to unmatched_customers.csv")

    df = df[df['CustomerKey'].notna()]

    # 🚛 4. Join with DimEquipment
    df = df.merge(dim_equipment[['EquipmentID', 'EquipmentKey']], on='EquipmentID', how='left')
    df = df[df['EquipmentKey'].notna()]

    # 📦 5. Join with DimFreightType
    df = df.merge(dim_freight[['FreightTypeID', 'FreightTypeKey']], on='FreightTypeID', how='left')
    df = df[df['FreightTypeKey'].notna()]

    # 🌍 6. Join with DimLocation (Origin & Destination)
    df = df.merge(dim_locations[['LocationID', 'LocationKey']],
                  left_on='OriginLocationID', right_on='LocationID', how='left') \
           .rename(columns={'LocationKey': 'OriginLocationKey'}) \
           .drop(columns=['LocationID'])

    df = df.merge(dim_locations[['LocationID', 'LocationKey']],
                  left_on='DestinationLocationID', right_on='LocationID', how='left') \
           .rename(columns={'LocationKey': 'DestinationLocationKey'}) \
           .drop(columns=['LocationID'])

    df = df[df['OriginLocationKey'].notna() & df['DestinationLocationKey'].notna()]

    # 🧹 7. Drop raw columns
    df.drop(columns=[
        'CustomerName', 'EquipmentID', 'FreightTypeID',
        'OriginLocationID', 'DestinationLocationID'
    ], inplace=True, errors='ignore')

    #  8. Remove duplicates
    dups = df[df.duplicated(subset=['ShipmentID'], keep=False)]
    if not dups.empty:
        dups.to_csv("duplicates_found.csv", index=False)
        print(f" Duplicates found: {len(dups)} rows saved to duplicates_found.csv")
    df = df.drop_duplicates(subset=['ShipmentID'])

    # 🛠️ 9. Fill nulls
    numeric_defaults = {
        'ShipmentValue': 0,
        'TotalVolume': 0,
        'TotalWeight': 0,
        'TotalPackages': 0,
        'IsDelivered': 0,
        'DurationDays': 0
    }
    df.fillna(numeric_defaults, inplace=True)

    # ⛓️ 10. Enforce FK types (as int)
    fk_columns = ['CustomerKey', 'EquipmentKey', 'FreightTypeKey', 'OriginLocationKey', 'DestinationLocationKey', 'DateKey']
    for col in fk_columns:
        df[col] = df[col].astype(int)

    print("✅ FactShipment transformation complete.")
    return df
