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
    data : pd.DataFrame
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
    complete_data = data.reindex(full_index) # Reindex the DataFrame to include all timestamps
    
    return complete_data
    
def remove_beyond_threshold(
    data: pd.DataFrame,
    horizontal_threshold: float,
    vertical_threshold: float,
    temperature_threshold: float
    ) -> pd.DataFrame:
    """
    Replaces values exceeding specific thresholds in the columns 'u', 'v', 'w', and 'T_s' with NaN.

    Parameters:
    - data: pd.DataFrame - input DataFrame with columns 'u', 'v', 'w', and 'T_s'
    - horizontal_threshold: float - threshold for 'u' and 'v'
    - vertical_threshold: float - threshold for 'w'
    - temperature_threshold: float - threshold for 'T_s'

    Returns:
    - pd.DataFrame - the cleaned DataFrame with outliers replaced by NaN
    """
    data_clean = data.copy()

    for col in ['u', 'v']:
        data_clean.loc[data_clean[col].abs() > horizontal_threshold, col] = np.nan

    data_clean.loc[data_clean['w'].abs() > vertical_threshold, 'w'] = np.nan

    data_clean.loc[data_clean['T_s'].abs() > temperature_threshold, 'T_s'] = np.nan

    return data_clean