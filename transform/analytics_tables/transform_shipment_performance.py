import pandas as pd

def transform_shipment_performance(df: pd.DataFrame) -> pd.DataFrame:
    print("Transforming ShipmentPerformanceMetrics with complete data...")
    print(f"Starting with {len(df)} rows")
    print(f"Input columns: {list(df.columns)}")

    if len(df) == 0:
        print("No rows to process!")
        return pd.DataFrame()

    # Calculate delivery variance
    df["DeliveryVariance"] = (df["ActualArrivalDate"] - df["PlannedArrivalDate"]).dt.days

    # Calculate on-time delivery flag
    df["OnTimeDeliveryFlag"] = df["DeliveryVariance"].apply(
        lambda x: 1 if pd.notnull(x) and x <= 0 else 0
    )

    # Handle DelayReason - preserve existing data or set based on delivery performance
    if "DelayReason" not in df.columns:
        # Create DelayReason based on delivery performance
        df["DelayReason"] = df["DeliveryVariance"].apply(
            lambda x: None if pd.isnull(x) or x <= 0 
                     else "Late Delivery" if x <= 7 
                     else "Significantly Delayed" if x <= 30 
                     else "Severely Delayed"
        )
        print("Created DelayReason based on delivery variance")
    else:
        print(f"Using existing DelayReason data. Null count: {df['DelayReason'].isnull().sum()}")

    # Handle CustomerSatisfactionScore more intelligently
    if "CustomerSatisfactionScore" not in df.columns:
        # Set satisfaction based on delivery performance
        def calculate_satisfaction(variance):
            if pd.isnull(variance):
                return 3  # Neutral for unknown
            elif variance <= -1:
                return 5  # Very satisfied (early delivery)
            elif variance <= 0:
                return 4  # Satisfied (on time)
            elif variance <= 3:
                return 3  # Neutral (slightly late)
            elif variance <= 7:
                return 2  # Dissatisfied (late)
            else:
                return 1  # Very dissatisfied (very late)
        
        df["CustomerSatisfactionScore"] = df["DeliveryVariance"].apply(calculate_satisfaction)
        print("Generated CustomerSatisfactionScore based on delivery performance")
    else:
        print(f"Using existing CustomerSatisfactionScore data. Null count: {df['CustomerSatisfactionScore'].isnull().sum()}")

    # Rename date columns to match target table
    df = df.rename(columns={
        "PlannedArrivalDate": "PlannedDeliveryDate",
        "ActualArrivalDate": "ActualDeliveryDate"
    })

    # Select final columns (adjust based on your actual table structure)
    final_columns = [
        "ShipmentKey",
        "PlannedDeliveryDate",
        "ActualDeliveryDate", 
        "DeliveryVariance",
        "OnTimeDeliveryFlag",
        "CustomerSatisfactionScore"
    ]
    
    # Add DelayReason if it exists in the DataFrame
    if "DelayReason" in df.columns:
        final_columns.append("DelayReason")
    
    df_final = df[final_columns].copy()

    print(f"Final columns: {list(df_final.columns)}")
    print(f"Sample DelayReason values: {df_final.get('DelayReason', pd.Series()).value_counts().head()}")
    print(f"CustomerSatisfactionScore distribution: {df_final['CustomerSatisfactionScore'].value_counts().sort_index()}")
    print("ShipmentPerformance transformation complete.")
    return df_final