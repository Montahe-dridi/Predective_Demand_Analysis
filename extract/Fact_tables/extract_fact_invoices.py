import pandas as pd
from sqlalchemy import text
from Configuration.db_config import get_source_engine,get_target_engine

def extract_purchasing_invoice():
    engine = get_source_engine()
    query = """
    SELECT
        ID AS InvoiceID,
        InvoiceDate,
        Supplier AS SupplierName,
        TotalTTCAmount AS TotalAmount,
        TotalTaxAmount AS TaxAmount,
        TotalNetAmount AS NetAmount,
        Statut AS PaymentStatus,
        DueDate AS PaymentDueDate,
        ID_PaymentTerms AS PaymentTermID
    FROM dbo.Purchasing_vwInvoice
    WHERE IsDeleted = 0 OR IsDeleted IS NULL
    """
    df = pd.read_sql(text(query), engine)
    df['InvoiceType'] = 'Purchase'
    print(f"✅ Extracted {len(df)} Purchasing Invoices")
    return df


def extract_sales_invoice():
    engine = get_source_engine()
    query = """
    SELECT
        ID AS InvoiceID,
        DateFacture AS InvoiceDate,
        RaisonSociale AS CustomerName,
        TotalTTC AS TotalAmount,  
        TotalTVA AS TaxAmount,
        TotalTTC AS NetAmount,
        ID_STATUS AS PaymentStatus,
        EcheanceDate AS PaymentDueDate,
        ID_NATURE AS PaymentTermID
    FROM dbo.Sales_vwFactures
    WHERE ID IS NOT NULL
    """
    df = pd.read_sql(text(query), engine)
    df['InvoiceType'] = 'Sales'
    print(f"✅ Extracted {len(df)} Sales Invoices")
    return df


def extract_dim_customer_keys():
    engine = get_target_engine()
    query = "SELECT CustomerKey, CustomerName  FROM DimCustomer WHERE IsCurrent = 1"
    return pd.read_sql(query, engine)

def extract_dim_supplier_keys():
    engine = get_target_engine()
    query = "SELECT SupplierKey,  SupplierName  FROM DimSupplier WHERE IsCurrent = 1"
    return pd.read_sql(query, engine)

def extract_dim_payment_term_keys():
    engine = get_target_engine()
    query = "SELECT PaymentTermKey, PaymentTermID,PaymentDays FROM DimPaymentTerm"
    return pd.read_sql(query, engine)

def extract_dim_dates_keys():
    engine = get_target_engine()  # Create the target engine here
    query = """
    SELECT DateKey, Date
    FROM DimDate
    """
    return pd.read_sql(query, engine)


def extract_fact_invoice():
    print("📦 Extracting purchasing and sales invoices...")

    # Extract raw invoices
    df_purchase = extract_purchasing_invoice()
    df_sales = extract_sales_invoice()
    
    # Combine
    df = pd.concat([df_purchase, df_sales], ignore_index=True)
    print(f"✅ Combined total invoices: {len(df)}")
    return df 

