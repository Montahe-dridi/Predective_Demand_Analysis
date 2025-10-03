import pandas as pd
from datetime import datetime
from sqlalchemy import text
from Configuration.db_config import get_target_engine


def generate_date_df(start_date, end_date):
    print(f"📅 Generating DimDate from {start_date.date()} to {end_date.date()}")
    dates = pd.date_range(start=start_date, end=end_date)
    
    df = pd.DataFrame()
    df['Date'] = dates
    df['DateKey'] = df['Date'].dt.strftime('%Y%m%d').astype(int)
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Quarter'] = df['Date'].dt.quarter
    df['DayOfWeek'] = df['Date'].dt.weekday + 1  # Monday = 1
    df['IsWeekend'] = df['DayOfWeek'].isin([6, 7]).astype(int)  # Saturday, Sunday
    df['IsHoliday'] = 0  # Static value (no holiday logic provided)
    df['Season'] = df['Month'].apply(lambda m: (
        'Winter' if m in [12, 1, 2] else
        'Spring' if m in [3, 4, 5] else
        'Summer' if m in [6, 7, 8] else
        'Autumn'
    ))
    df['FiscalPeriod'] = df['Year'].astype(str) + '-P' + df['Month'].astype(str).str.zfill(2)

    print(f"✅ Generated {len(df)} date records.")
    return df


def load_dim_date(df):
    print("🚚 Loading data into DimDate...")
    engine = get_target_engine()
    table_name = "DimDate"

    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {table_name}"))
        print("🗑️ Cleared existing data from DimDate")

    df.to_sql(name=table_name, con=engine, if_exists='append', index=False)
    print(f"✅ Successfully loaded {len(df)} records into DimDate")


def run_pipeline():
    try:
        start_date = datetime(2000, 1, 1)
        end_date = datetime(2099, 12, 31)
        df = generate_date_df(start_date, end_date)
        load_dim_date(df)
        print("🎉 DimDate pipeline completed successfully!")
    except Exception as e:
        print(f"❌ DimDate pipeline failed: {e}")
