# utils/scd_helpers.py

import pandas as pd

def apply_effective_date(df, source_date_col='LastModifiedOnDate', fallback_col='CreatedOnDate', default_date='2000-01-01'):
    """
    Applies EffectiveDate logic based on source column with fallback and default.
    
    Args:
        df (DataFrame): Your transformed dataframe
        source_date_col (str): Column to use as primary effective date
        fallback_col (str): Fallback column if source_date_col is null or missing
        default_date (str): Final fallback if both are missing/null
        
    Returns:
        DataFrame: same dataframe with EffectiveDate column added
    """
    df = df.copy()

    if source_date_col not in df.columns:
        df[source_date_col] = pd.NaT

    if fallback_col not in df.columns:
        df[fallback_col] = pd.NaT

    df['EffectiveDate'] = pd.to_datetime(df[source_date_col], errors='coerce')
    
    # If EffectiveDate is missing, try fallback
    fallback_mask = df['EffectiveDate'].isna()
    df.loc[fallback_mask, 'EffectiveDate'] = pd.to_datetime(df.loc[fallback_mask, fallback_col], errors='coerce')

    # Final fallback
    df['EffectiveDate'] = df['EffectiveDate'].fillna(pd.to_datetime(default_date))

    return df
