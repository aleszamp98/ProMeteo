import pandas as pd
import numpy as np
from typing import Tuple
import core


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

def linear_interp(left_border, right_border, n_points):
    return interp_points

def identify_interp_spikes(beyond_bounds_mask, max_consecutive_spikes):
    return array_modified, spike_flag

def identify_interp_spikes(array, mask, n_max_spike):

    dum=False
    count_spike_day=0
    count_spike=0
    N=len(array)

    for i in range(N):
        if mask[i]:
            if dum==False:
                spike_index=[i] #mi salvo le posizioni delle spike, la lunghezza di questo array definisce quante spike si susseguono
                dum=True
            else:
                spike_index.append(i)
        else:
            if dum==True:
                M=len(spike_index)
                if M<=n_max_spike:
                    #interpolazione lineare
                    y_0=array[spike_index[0]-1] #valore primo vicino a sinistra
                    y_1=array[spike_index[-1]+1] #valore primo vicino a destra
                    diff=y_1-y_0
                    x=np.arange(1, M+1) #(1,2,3)
                    array[spike_index]=y_0+x*diff/(M+1)                  
                    dum=False #azzero il flag del contatore
                    count_spike=count_spike+1
                    count_spike_day=count_spike_day+1
                else:
                    dum=False #azzero il flag del contatore
                    del spike_index
    return array, count_spike


def despiking_VM97(array_to_despike : np.ndarray,
                   c : float,
                   window_length : int,
                   max_consecutive_spikes : int,
                   max_iterations : int
                   ) -> np.ndarray :
    spike_flag = True
    iteration = 0
    c_increment = 0.1
    while spike_flag and iteration < max_iterations:
        running_mean, running_std = core.running_stats(array_to_despike, 
                                                  window_length)
        upper_bound = running_mean+c*running_std
        lower_bound = running_mean-c*running_std
        beyond_bounds_mask = (array_to_despike > upper_bound) | (array_to_despike < lower_bound)
        array_despiked, spike_flag = identify_interp_spikes(array_to_despike,
                                                            beyond_bounds_mask, 
                                                            max_consecutive_spikes)
        c += c_increment # at each iteration the bounds increment their distance 
        iteration += 1
    return array_despiked


def despiking_ROBUST(data : pd.DataFrame) -> pd.DataFrame:

    data_despiked=data


    return data_despiked

