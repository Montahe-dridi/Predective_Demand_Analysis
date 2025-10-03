import pandas as pd
import unicodedata
import re

def normalize_text(text):
    """Normalize text by removing accents, punctuation, and extra spaces; lowercasing."""
    if pd.isna(text):
        return None
    text = str(text).strip()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(r'[^\w\s]', ' ', text)   # Remove punctuation
    text = re.sub(r'\s+', ' ', text)       # Collapse spaces
    return text.lower().strip()


def transform_fact_invoice(df, dim_dates, dim_customers, dim_suppliers, dim_payment_terms):
    print("DimDates columns:", dim_dates.columns)
    print("🔄 Transforming FactInvoice data...")

    # --- Normalize InvoiceDate ---
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']).dt.normalize()
    dim_dates['Date'] = pd.to_datetime(dim_dates['Date']).dt.normalize()

    # --- Normalize names ---
    dim_customers['CustomerName_norm'] = dim_customers['CustomerName'].apply(normalize_text)
    dim_suppliers['SupplierName_norm'] = dim_suppliers['SupplierName'].apply(normalize_text)

    if 'CustomerName' in df.columns:
        df['CustomerName_norm'] = df['CustomerName'].apply(normalize_text)
    if 'SupplierName' in df.columns:
        df['SupplierName_norm'] = df['SupplierName'].apply(normalize_text)

    # --- Merge DateKey ---
    df = df.merge(
        dim_dates.rename(columns={'Date': 'InvoiceDate', 'DateKey': 'DateKey'}),
        on='InvoiceDate',
        how='left'
    )
    before = len(df)
    df = df[df['DateKey'].notna()]
    print(f"✅ Kept {len(df)} rows with valid DateKeys (removed {before - len(df)})")

    # --- Split Sales vs Purchase ---
    sales_mask = df['InvoiceType'] == 'Sales'
    purchase_mask = df['InvoiceType'] == 'Purchase'

    # --- Merge CustomerKey for Sales ---
    sales_df = df[sales_mask].copy()
    sales_df = sales_df.merge(
        dim_customers[['CustomerKey', 'CustomerName_norm']],
        on='CustomerName_norm',
        how='left'
    )
    unmatched_customers = sales_df[sales_df['CustomerKey'].isna()][['CustomerName_norm']].drop_duplicates()
    if not unmatched_customers.empty:
        unmatched_customers.to_csv("unmatched_customers.csv", index=False)
        print(f"❌ {len(unmatched_customers)} unmatched sales customers written to 'unmatched_customers.csv'")

    # --- Merge SupplierKey for Purchases ---
    purchase_df = df[purchase_mask].copy()
    purchase_df = purchase_df.merge(
        dim_suppliers[['SupplierKey', 'SupplierName_norm']],
        on='SupplierName_norm',
        how='left'
    )
    unmatched_suppliers = purchase_df[purchase_df['SupplierKey'].isna()][['SupplierName_norm']].drop_duplicates()
    if not unmatched_suppliers.empty:
        unmatched_suppliers.to_csv("unmatched_suppliers.csv", index=False)
        print(f"❌ {len(unmatched_suppliers)} unmatched purchase suppliers written to 'unmatched_suppliers.csv'")

    # --- Combine back ---
    df = pd.concat([sales_df, purchase_df], ignore_index=True)

    # --- Drop unmatched FK rows ---
    before = len(df)
    df = df[~((df['InvoiceType'] == 'Sales') & df['CustomerKey'].isna())]
    df = df[~((df['InvoiceType'] == 'Purchase') & df['SupplierKey'].isna())]
    print(f"✅ Kept {len(df)} rows after removing unmatched customers/suppliers (removed {before - len(df)})")

    # --- Normalize PaymentTermID ---
    df['PaymentTermID'] = pd.to_numeric(df['PaymentTermID'], errors='coerce').astype('Int64')
    dim_payment_terms['PaymentTermID'] = pd.to_numeric(dim_payment_terms['PaymentTermID'], errors='coerce').astype('Int64')

    print("🔎 Source FactInvoice PaymentTermID values:", df['PaymentTermID'].dropna().unique())
    print("🔎 DimPaymentTerm PaymentTermID values:", dim_payment_terms['PaymentTermID'].dropna().unique())

    # --- Merge PaymentTerm ---
    df = df.merge(
        dim_payment_terms[['PaymentTermKey', 'PaymentTermID']],
        on='PaymentTermID',
        how='left'
    )

    # --- Assign dummy key 0 for unknowns ---
    missing_terms = df['PaymentTermKey'].isna().sum()
    if missing_terms > 0:
        df['PaymentTermKey'] = df['PaymentTermKey'].fillna(0).astype(int)
        print(f"⚠️ {missing_terms} invoices had unknown PaymentTermID → assigned to dummy key 0")

    # --- Fix NetAmount where Tax=0 and Net=0 but Total>0 ---
    mask_fix = (df['TaxAmount'] == 0) & (df['NetAmount'] == 0) & (df['TotalAmount'] != 0)
    df.loc[mask_fix, 'NetAmount'] = df.loc[mask_fix, 'TotalAmount']
    if mask_fix.sum() > 0:
        print(f"🔧 Corrected NetAmount for {mask_fix.sum()} rows where Tax=0 and Net=0.")

    # --- Ensure dummy key logic ---
    df.loc[df['InvoiceType'] == 'Sales', 'SupplierKey'] = df.loc[df['InvoiceType'] == 'Sales', 'SupplierKey'].fillna(0).astype(int)
    df.loc[df['InvoiceType'] == 'Purchase', 'CustomerKey'] = df.loc[df['InvoiceType'] == 'Purchase', 'CustomerKey'].fillna(0).astype(int)

    # --- Map PaymentStatus (text → int) ---
    status_map = {
        'Validée': 1,
        'Payée': 2,
        'En attente': 3,
        'Annulée': 4
    }
    df['PaymentStatus'] = df['PaymentStatus'].map(status_map).fillna(df['PaymentStatus'])

    # --- CRITICAL FIX: Handle PaymentDueDate nulls ---
    if 'PaymentDueDate' in df.columns:
        # Convert to datetime first
        df['PaymentDueDate'] = pd.to_datetime(df['PaymentDueDate'], errors='coerce')
        
        # Count nulls before handling
        null_count = df['PaymentDueDate'].isna().sum()
        if null_count > 0:
            print(f"⚠️ Found {null_count} NULL PaymentDueDate values")
            
            # Merge PaymentDays from DimPaymentTerm for calculation
            df = df.merge(
                dim_payment_terms[['PaymentTermKey', 'PaymentDays']],
                on='PaymentTermKey',
                how='left'
            )
            
            # Calculate PaymentDueDate = InvoiceDate + PaymentDays for NULL values
            mask_null_due_date = df['PaymentDueDate'].isna()
            df.loc[mask_null_due_date, 'PaymentDueDate'] = (
                df.loc[mask_null_due_date, 'InvoiceDate'] + 
                pd.to_timedelta(df.loc[mask_null_due_date, 'PaymentDays'].fillna(30), unit='D')
            )
            print(f" Calculated PaymentDueDate using InvoiceDate + PaymentDays for {null_count} rows")
            
            # Drop the temporary PaymentDays column
            df = df.drop(columns=['PaymentDays'])
        
        # Final check - ensure no nulls remain
        remaining_nulls = df['PaymentDueDate'].isna().sum()
        if remaining_nulls > 0:
            print(f"❌ Still have {remaining_nulls} NULL PaymentDueDate values after fix!")
        else:
            print("✅ All PaymentDueDate values are now non-null")

    # --- Remove duplicate InvoiceIDs (keep latest by InvoiceDate) ---
    before = len(df)
    df = df.sort_values(by=['InvoiceID', 'InvoiceDate']).drop_duplicates(subset=['InvoiceID'], keep='last')
    print(f"✅ Removed {before - len(df)} duplicate InvoiceIDs (kept {len(df)})")

    # --- Final selection ---
    fact_invoice = df[[
        'InvoiceID', 'InvoiceDate', 'DateKey', 'CustomerKey', 'SupplierKey',
        'TotalAmount', 'TaxAmount', 'NetAmount', 'PaymentStatus',
        'PaymentDueDate', 'PaymentTermKey', 'InvoiceType'
    ]].copy()

    print("✅ FactInvoice transformation complete.")
    return fact_invoice