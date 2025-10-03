def run_pipeline():
    from extract.Dim_tables.extract_paymentterm import extract_payment_term_data
    from transform.Dim_tables.transform_paymentterm import transform_payment_term_data
    from validation.validate_data_quality import validate_payment_term_data
    from load.Dim_tables.load_dim_paymentterm import load_dim_payment_term

    df_raw = extract_payment_term_data()
    df_clean = transform_payment_term_data(df_raw)
    validate_payment_term_data(df_clean)
    load_dim_payment_term(df_clean)

if __name__ == "__main__":
    run_pipeline()