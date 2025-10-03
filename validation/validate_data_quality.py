import pandas as pd
# tralis_etl/validation/validate_data_quality.py

def validate_dim_product(df):
    print("🔍 Validating DimProduct data...")

    if df['ProductID'].isnull().any():
        raise ValueError("❌ ProductID contains nulls!")

    if df['ProductName'].eq('').any():
        raise ValueError("❌ ProductName contains empty strings!")

    if df.duplicated(subset='ProductID').any():
        raise ValueError("❌ Duplicate ProductIDs found!")

    print("✅ DimProduct data passed validation.")

def validate_customer_data(df):
    print("🔍 Validating DimCustomer data...")

    required_columns = [
        'CustomerID', 'CustomerName', 'CustomerType', 'CustomerCategory',
        'CustomerCountry', 'CustomerCity', 'CustomerRegion',
        'CustomerSegment', 'IsActive','LastModifiedOnDate',
    ]

    # Check required columns exist
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"❌ Missing required column: {col}")

    # Check for nulls in essential columns
    for col in ['CustomerID', 'CustomerName', 'CustomerCountry']:
        if df[col].isnull().any():
            raise ValueError(f"❌ Null values found in required column: {col}")

    # Optional: Warn if duplicates exist (but don’t raise error)
    dupes = df[df.duplicated(subset=['CustomerID'], keep=False)]
    if not dupes.empty:
        print(f"⚠️ Warning: {len(dupes)} duplicate CustomerID records found. Proceeding anyway.")

    print("✅ DimCustomer data passed validation.")


def validate_supplier_data(df):
    print("🔍 Validating DimSupplier data...")

    required_columns = [
        'SupplierID', 'SupplierName', 'SupplierType', 
        'SupplierCountry', 'SupplierCity', 'SupplierRegion',
         'IsActive', 'LastModifiedOnDate'
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"❌ Missing required column: {col}")

    for col in ['SupplierID', 'SupplierName', 'SupplierCountry']:
        if df[col].isnull().any():
            raise ValueError(f"❌ Null values found in required column: {col}")

    dupes = df[df.duplicated(subset=['SupplierID'], keep=False)]
    if not dupes.empty:
        print(f"⚠️ Warning: {len(dupes)} duplicate SupplierID records found. Proceeding anyway.")

    print("✅ DimSupplier data passed validation.")


# tralis_etl/validation/validate_location_data.py

def validate_location_data(df):
    print("🔍 Validating DimLocation data...")

    required_columns = [
        'LocationID', 'LocationName', 'LocationType',
         'IsActive', 'CreatedOnDate'
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"❌ Missing required column: {col}")

    for col in ['LocationID', 'LocationName']:
        if df[col].isnull().any():
            raise ValueError(f"❌ Null values found in required column: {col}")

    dupes = df[df.duplicated(subset=['LocationID'], keep=False)]
    if not dupes.empty:
        print(f"⚠️ Warning: {len(dupes)} duplicate LocationID records found. Proceeding anyway.")

    print("✅ DimLocation data passed validation.")

# tralis_etl/validation/validate_data_quality.py

def validate_equipment_data(df):
    print("🔍 Validating DimEquipment data...")

    required_columns = [
        'EquipmentID', 'EquipmentType', 'OperationType',
        'IsActive', 'EffectiveDate'
    ]

    # 1. Check required columns
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"❌ Missing required column: {col}")

    # 2. Null ID check
    if df['EquipmentID'].isnull().any():
        raise ValueError("❌ Null EquipmentID values found!")

    # 3. Duplicates
    duplicates = df.duplicated(subset=['EquipmentID']).sum()
    if duplicates > 0:
        print(f"⚠️ Warning: {duplicates} duplicate EquipmentID records found.")

    
    # 5. EffectiveDate must be a valid datetime and not null
    if df['EffectiveDate'].isnull().any():
        print("⚠️ Warning: Some EffectiveDate values are missing. Consider setting defaults or cleaning source data.")

    # 6. Optional: Validate known designations (optional)
    known_types = ['Camion', 'Tracteur', 'Remorque', 'Semi-remorque', 'Unknown']
    unknown_types = df[~df['EquipmentType'].isin(known_types)]['EquipmentType'].unique()
    if len(unknown_types) > 0:
        print(f"⚠️ Warning: Found unknown EquipmentType values: {unknown_types}")

    print("✅ DimEquipment data passed validation.")


# freight_type

def validate_freight_type_data(df):
    print("🔍 Validating DimFreightType data...")

    if df['FreightTypeID'].isnull().any():
        raise ValueError("❌ FreightTypeID contains nulls")

    if df['FreightTypeID'].duplicated().any():
        print("⚠️ Warning: Duplicate FreightTypeID records found")

    

    print("✅ DimFreightType data passed validation.")



def validate_payment_term_data(df):
    print("🔍 Validating DimPaymentTerm data...")

    # Duplicates check
    if df['PaymentTermID'].duplicated().any():
        raise ValueError("❌ Duplicate PaymentTermID values found in DimPaymentTerm")

    # Required columns
    required_cols = ['PaymentTermID', 'PaymentTermName', 'PaymentDays']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ Missing required columns in DimPaymentTerm: {missing_cols}")

    print("✅ DimPaymentTerm validation passed")




def validate_dim_date_data(df):
    print("🔍 Validating DimDate data...")
    if df.empty:
        raise ValueError("❌ DimDate is empty.")
    if df['DateKey'].duplicated().any():
        raise ValueError("❌ Duplicate DateKey values found.")
    if df['Date'].isnull().any():
        raise ValueError("❌ Some Date values are null.")
    print("✅ DimDate passed validation.")

#----------------------------------Fact_tables----------------------------------------------------------------#
#Shipment_table
def validate_fact_shipment(df):
    print("🔍 Validating FactShipment data...")

    required_columns = [
        'ShipmentID', 'DateKey', 'CustomerKey', 
        'EquipmentKey', 'FreightTypeKey', 'OriginLocationKey', 'DestinationLocationKey',
        'ShipmentDate', 'PlannedArrivalDate', 'ActualDepartureDate', 'ActualArrivalDate',
        'TotalWeight', 'TotalVolume', 'TotalPackages', 'ShipmentValue',
         'IsDelivered', 'DurationDays'
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"❌ Missing required column: {col}")

    if df.isnull().any().any():
        nulls = df.isnull().sum()
        problematic_cols = nulls[nulls > 0]
        print(f"⚠️ Warning: Found nulls in columns:\n{problematic_cols}")
    else:
        print("✅ No nulls found.")

    if df.duplicated(subset=['ShipmentID']).any():
        raise ValueError("❌ Duplicate ShipmentID values found!")

    print("✅ FactShipment data passed validation.")


import pandas as pd

def validate_fact_invoice_data(df):
    print("🔍 Validating FactInvoice data...")

    # 1. Null checks
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        print("⚠️ Warning: Found nulls in columns:")
        print(nulls)
    else:
        print("✅ No null values found.")

    # 2. Duplicate InvoiceID
    duplicates = df[df['InvoiceID'].duplicated()]
    if not duplicates.empty:
        print(f"⚠️ Found {len(duplicates)} duplicate InvoiceIDs. Saved to duplicate_invoices.csv.")
        duplicates.to_csv("duplicate_invoices.csv", index=False)
    else:
        print("✅ No duplicate InvoiceIDs found.")

    # 3. Ensure int type for PaymentStatus
    if not pd.api.types.is_integer_dtype(df['PaymentStatus']):
        print("❌ PaymentStatus is not integer dtype, fixing...")
        df['PaymentStatus'] = pd.to_numeric(df['PaymentStatus'], errors='coerce').fillna(0).astype(int)
    else:
        print("✅ PaymentStatus is valid integer dtype.")

    # 4. Dummy keys validation
    invalid_sales = df[(df['InvoiceType'] == 'Sales') & (df['SupplierKey'] != 0)]
    invalid_purchases = df[(df['InvoiceType'] == 'Purchase') & (df['CustomerKey'] != 0)]
    if invalid_sales.empty and invalid_purchases.empty:
        print("✅ Dummy key validation passed (Sales → Supplier=0, Purchases → Customer=0).")
    else:
        print("❌ Dummy key validation failed. Found rows:")
        if not invalid_sales.empty:
            print(f"  - {len(invalid_sales)} Sales rows with non-0 SupplierKey")
        if not invalid_purchases.empty:
            print(f"  - {len(invalid_purchases)} Purchase rows with non-0 CustomerKey")

    print("✅ Validation complete.")
    return df
#-----------------------------Analytics_tables-----------------------------#


import pandas as pd

def validate_shipment_performance_data(df: pd.DataFrame) -> bool:
    print("🔍 Validating ShipmentPerformanceMetrics data...")

    issues_found = False
    warnings_found = False

    # --- Check for critical nulls (ShipmentKey should never be null) ---
    if df["ShipmentKey"].isnull().any():
        print("❌ CRITICAL: ShipmentKey has null values!")
        issues_found = True
    else:
        print("✅ ShipmentKey has no null values.")

    # --- Check for data quality issues (warnings, not necessarily blockers) ---
    planned_nulls = df["PlannedDeliveryDate"].isnull().sum()
    actual_nulls = df["ActualDeliveryDate"].isnull().sum()
    variance_nulls = df["DeliveryVariance"].isnull().sum()
    
    if planned_nulls > 0:
        print(f"⚠️ WARNING: {planned_nulls} rows with null PlannedDeliveryDate")
        warnings_found = True
        
    if actual_nulls > 0:
        print(f"⚠️ WARNING: {actual_nulls} rows with null ActualDeliveryDate")
        print(f"   This affects {actual_nulls/len(df)*100:.1f}% of records")
        warnings_found = True
        
    if variance_nulls > 0:
        print(f"⚠️ WARNING: {variance_nulls} rows with null DeliveryVariance")
        warnings_found = True

    # --- Check for duplicates on ShipmentKey ---
    if df["ShipmentKey"].duplicated().any():
        print(f"❌ Found {df['ShipmentKey'].duplicated().sum()} duplicate ShipmentKeys.")
        issues_found = True
    else:
        print("✅ No duplicate ShipmentKeys found.")

    # --- DeliveryVariance reasonableness (only for non-null values) ---
    if "DeliveryVariance" in df.columns:
        non_null_variance = df["DeliveryVariance"].notna()
        if non_null_variance.any():
            unreasonable = df[non_null_variance & (df["DeliveryVariance"].abs() > 365)]
            if not unreasonable.empty:
                print(f"⚠️ WARNING: {len(unreasonable)} rows with unreasonable DeliveryVariance (>365 days).")
                warnings_found = True

    # --- OnTimeDeliveryFlag must be 0 or 1 ---
    if not df["OnTimeDeliveryFlag"].isin([0, 1]).all():
        print("❌ OnTimeDeliveryFlag has invalid values (not 0/1).")
        issues_found = True
    else:
        print("✅ OnTimeDeliveryFlag valid (0/1 only).")

    # --- CustomerSatisfactionScore must be between 1 and 5 ---
    if not df["CustomerSatisfactionScore"].between(1, 5).all():
        print("❌ CustomerSatisfactionScore has values outside 1–5 range.")
        issues_found = True
    else:
        print("✅ CustomerSatisfactionScore valid (1–5).")

    # Final validation result
    if issues_found:
        print("❌ Validation completed with CRITICAL issues.")
        return False
    elif warnings_found:
        print("⚠️ Validation completed with warnings. Data quality issues noted but proceeding...")
        return True
    else:
        print("✅ Validation passed with no issues.")
        return True
    


def validate_freight_cost(df: pd.DataFrame):
    print("🔍 Validating FreightCostAnalysis data...")

    issues = False

    # Check nulls
    nulls = df.isnull().sum()
    if nulls.any():
        print(f"⚠️ Found nulls in columns:\n{nulls[nulls > 0]}")
        issues = True

    # Negative costs
    negatives = df[(df["PlannedFreightCost"] < 0) | (df["ActualFreightCost"] < 0)]
    if not negatives.empty:
        print(f"⚠️ {len(negatives)} rows with negative costs.")
        issues = True

    # CostEfficiency sanity check
    if (df["CostEfficiencyRatio"] < 0).any():
        print("⚠️ Invalid efficiency ratios (<0).")
        issues = True

    if not issues:
        print("✅ Validation passed with no issues.")
    else:
        print("❌ Validation completed with issues.")
