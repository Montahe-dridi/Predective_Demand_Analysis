# Pipelines/run_dim_freight_type.py

from extract.Dim_tables.extract_freighttype import extract_freight_type_data
from transform.Dim_tables.transform_freighttype import transform_freight_type_data
from validation.validate_data_quality import validate_freight_type_data
from load.Dim_tables.load_dim_freighttype import load_dim_freight_type

def run_pipeline():
    df_raw = extract_freight_type_data()
    df_clean = transform_freight_type_data(df_raw)
    validate_freight_type_data(df_clean)
    load_dim_freight_type(df_clean)

if __name__ == "__main__":
    run_pipeline()