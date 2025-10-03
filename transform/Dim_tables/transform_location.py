
import pandas as pd

def transform_location_data(df):
    print("🔄 Transforming location data...")
    df = df.copy()
    df['LocationID'] = df['LocationID'].astype(str).str.strip()
    df['LocationName'] = df['LocationName'].fillna('Unknown').str.title()
    df['LocationType'] = df['LocationType'].fillna('Unknown')
    df['IsActive'] = df['IsActive'].apply(lambda x: True if x == 1 else False)
    print(f"📊 Transformed {len(df)} location records")

    df = df.drop_duplicates(subset=['LocationID'], keep='last')

    

    return df[[
        'LocationID', 'LocationName', 'LocationType',
         'IsActive', 'CreatedOnDate'
    ]]
