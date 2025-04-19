import pandas as pd
import numpy as np
from typing import Tuple


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
    Replaces values exceeding specific thresholds in the columns "u", "v", "w", and "T_s" with NaN.
    The threshold definition is

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with columns "u", "v", "w", and "T_s" and datetime index.
    horizontal_threshold : float
        Threshold for horizontal components of the wind vector: "u" and "v".
    vertical_threshold : float
        Threshold for the vertical component of the wind vector: "w".
    temperature_threshold : float
        Threshold for sonic temperature "T_s".

    Returns:
    --------
    - pd.DataFrame
        The cleaned DataFrame with outliers replaced by NaN.

    Notes:
    ------
    The distinction of thresholds into horizontal and vertical is based on the fact that
    horizontal motions are usually more intense than vertical motions by two or more
    orders of magnitude.
    """
    data_clean = data.copy()

    for col in ['u', 'v']:
        data_clean.loc[data_clean[col].abs() > horizontal_threshold, col] = np.nan

    data_clean.loc[data_clean['w'].abs() > vertical_threshold, 'w'] = np.nan

    data_clean.loc[data_clean['T_s'].abs() > temperature_threshold, 'T_s'] = np.nan

    return data_clean

##### DESPIKING SECTION #####
# definizione input: finestra in min per calcolo mobile, numero di valori consecutivi fuori soglia, costanti da cui partire:
#  c_h (orizzontale), c_v (verticale), c_T (temp)

# conviene trasferirla nel modulo core (perchè la media mobile è usata anche da reynolds module)
def running_stats(array : np.ndarray,
                  window_length : int) -> Tuple[np.ndarray, np.ndarray]:
    
    return running_mean, running_std

def linear_interp(left_border, right_border, n_points):
    return interp_points

def identify_interp_spikes(beyond_bounds_mask, max_consecutive_spikes):
    return array_modified, spike_flag

def data_despiking_VM97(data: pd.DataFrame,
                        window_length : int,
                        max_consecutive_spikes : int,
                        max_iterations : int,
                        c_h: float,
                        c_v: float,
                        c_T: float,
                        ) -> pd.DataFrame:
    data_despiked = pd.DataFrame(index=data.index, columns=data.columns)
    c_list = [c_h, c_h, c_v, c_T]
    c_increment=0.1

    for col, c in zip(['u', 'v', 'w', 'T_s'], c_list):
        array_to_despike = data[col].to_numpy()
        spike_flag = True
        iteration = 0
        while spike_flag and iteration < max_iterations:
            running_mean, running_std = running_stats(array_to_despike, window_length)
            upper_bound = running_mean+c*running_std
            lower_bound = running_mean-c*running_std
            beyond_bounds_mask = (array_to_despike > upper_bound) | (array_to_despike < lower_bound)
            array_despiked, spike_flag = identify_interp_spikes(beyond_bounds_mask, max_consecutive_spikes)
            c += c_increment # at each iteration the threshold increment their distance 
            iteration += 1
            del running_mean, running_std, upper_bound, lower_bound, beyond_bounds_mask

        data_despiked[col] = array_despiked
        del array_despiked

    return data_despiked



def data_despiking_ROBUST(data : pd.DataFrame) -> pd.DataFrame:

    data_despiked=data


    return data_despiked