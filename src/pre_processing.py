import pandas as pd
import numpy as np


## controllo se possiede un numero di entrate pari alla frequenza* delta_t compreso tra l'inzio e la fine del 
def fill_missing_timestamps(data: pd.DataFrame, freq: float) -> pd.DataFrame:
    """
    Ensures a DataFrame with a datetime index includes all timestamps between 
    the first and last entry, based on the given frequency. Missing timestamps 
    are added with NaN values for all columns.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with a datetime index.
    freq : float
        Sampling frequency in Hertz (Hz).

    Returns:
    --------
    pd.DataFrame
        DataFrame reindexed to include all expected timestamps.
    """
    if freq <= 0:
        raise ValueError("Frequency (Hz) must be a positive number.")
    dt_s=1/freq
    freq = pd.to_timedelta(dt_s, unit='s') # pandas-compatible frequency string
    data = data.sort_index()    
    full_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq=freq) # Create the complete range of timestamps
    complete_df = data.reindex(full_index) # Reindex the DataFrame to include all timestamps
    
    return complete_df