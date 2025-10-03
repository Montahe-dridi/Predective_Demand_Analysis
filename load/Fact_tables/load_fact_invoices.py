from sqlalchemy import Integer, Float, DateTime
from Configuration.db_config import get_target_engine
from sqlalchemy import String

...



def load_fact_invoice(df):
    engine = get_target_engine()
    print(f"📥 Loading {len(df)} rows into FactInvoice...")

    df.to_sql(
        name='FactInvoices',
        con=engine,
        if_exists='append',
        index=False,
        dtype={
            'InvoiceID': Integer(),
            'DateKey': Integer(),
            'CustomerKey': Integer(),
            'SupplierKey': Integer(),
            'PaymentTermKey': Integer(),
            'TotalAmount': Float(),
            'TaxAmount': Float(),
            'NetAmount': Float(),
            'PaymentStatus': Integer(),
            'PaymentDueDate': DateTime(),
            'InvoiceDate': DateTime(),
            'InvoiceType': String(length=10)  
        }
    )

    print("✅ FactInvoice load complete.")
