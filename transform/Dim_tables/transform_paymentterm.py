import pandas as pd

def transform_payment_term_data(df):
    print("🔄 Transforming payment term data...")

    df = df.copy()

    # Clean string columns
    df['PaymentTermID'] = df['PaymentTermID'].astype(str).str.strip()
    df['PaymentTermName'] = df['PaymentTermName'].astype(str).str.strip()

    # Ensure PaymentDays is numeric (handle bad data gracefully)
    df['PaymentDays'] = pd.to_numeric(df['PaymentDays'], errors='coerce').fillna(0).astype(int)

    # Add technical columns if needed
    df['IsActive'] = df['IsActive'].fillna(1).astype(int)

    print(f"📊 Transformed {len(df)} payment term records")
    return df
