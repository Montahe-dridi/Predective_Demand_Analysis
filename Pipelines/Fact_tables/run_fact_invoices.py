import pandas as pd
from extract.Fact_tables.extract_fact_invoices import (
    extract_fact_invoice,
    extract_dim_customer_keys,
    extract_dim_supplier_keys,
    extract_dim_payment_term_keys,
    extract_dim_dates_keys
)
from transform.Fact_tables.transform_fact_invoices import transform_fact_invoice
from load.Fact_tables.load_fact_invoices import load_fact_invoice
from validation.validate_data_quality import validate_fact_invoice_data

def run_pipeline():
    print("🧾 Starting FactInvoice pipeline")

    # 1️⃣ Extract
    df = extract_fact_invoice()
    dim_dates = extract_dim_dates_keys()
    dim_customers = extract_dim_customer_keys()
    dim_suppliers = extract_dim_supplier_keys()
    dim_payment_terms = extract_dim_payment_term_keys()

    print(f"✅ Extracted {len(df)} raw invoice records")

    # 2️⃣ Transform
    df_transformed = transform_fact_invoice(
        df,
        dim_dates,
        dim_customers,
        dim_suppliers,
        dim_payment_terms
    )

    # 3️⃣ Validate
    validate_fact_invoice_data(df_transformed)

    # 4️⃣ Load
    load_fact_invoice(df_transformed)

def run_fact_invoices():
    run_pipeline()
