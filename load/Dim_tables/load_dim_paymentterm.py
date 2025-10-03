def load_dim_payment_term(df):
    import pandas as pd
    from sqlalchemy import create_engine, String, Boolean, DateTime
    from Configuration.db_config import get_target_engine

    print("🚚 Loading into DimPaymentTerm ...")
    engine = get_target_engine()

    df.to_sql(
        'DimPaymentTerm',
        con=engine,
        if_exists='append',
        index=False,
        dtype={
            'PaymentTermID': String(),
            'PaymentTermName': String(),
            'IsActive': Boolean(),
            
        }
    )
    print(f"✅ Successfully loaded {len(df)} records into DimPaymentTerm.")
