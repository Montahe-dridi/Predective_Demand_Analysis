from sqlalchemy import Integer, Float, DateTime, Boolean
from Configuration.db_config import get_target_engine


def load_fact_shipment(df):
    engine = get_target_engine()
    print(f"🚛 Loading {len(df)} rows into FactShipment...")

    df.to_sql(
        name='FactShipments',
        con=engine,
        if_exists='append',
        index=False,
        
        dtype={
            'ShipmentID': Integer(),
            'DateKey': Integer(),
            'CustomerKey': Integer(),
            'EquipmentKey': Integer(),
            'FreightTypeKey': Integer(),
            'OriginLocationKey': Integer(),
            'DestinationLocationKey': Integer(),
            'ShipmentDate': DateTime(),
            'PlannedArrivalDate': DateTime(),
            'ActualDepartureDate': DateTime(),
            'ActualArrivalDate': DateTime(),
            'TotalWeight': Float(),
            'TotalVolume': Float(),
            'TotalPackages': Integer(),
            'ShipmentValue': Float(),
            'DurationDays': Integer(),
            'IsDelivered': Boolean()
            
        }
    )

    print("✅ FactShipment load complete.")
