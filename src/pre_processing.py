import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional
from . import core


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

def linear_interp(left_value : float,
                  right_value : float,
                  length : int
                  ) -> np.ndarray:
    """
    Performs linear interpolation between two values (`left_value` and `right_value`)
    over an array of specified length (`length`).
    
    Parameters
    ----------
    left_value : float
        The reference value on the left border.
    right_value : float
        The reference value on the right border.
    length : int
        The length of the array to perform the interpolation over.
    
    Returns
    -------
    np.ndarray
        An array containing the interpolated values.
    """
    # Create an array of relative indices for the length
    x = np.arange(1, length + 1)
    
    # Compute the interpolation
    interpolated_values = left_value + (x / (length + 1)) * (right_value - left_value)
    
    return interpolated_values

def identify_interp_spikes(array: np.ndarray,
                           mask: np.ndarray,
                           max_length_spike: int
                           ) -> tuple[np.ndarray, int]:
    """
    Identifies and interpolates spikes in the provided array based on the given mask.
    
    A spike is defined as a sequence of consecutive True values in the mask that is shorter than or equal to
    the specified maximum length (`max_length_spike`). The function applies linear interpolation to replace 
    the spike values with interpolated ones. If the spike is at the boundary of the array (either at the start
    or at the end), interpolation is not performed.

    Parameters:
    ----------
    array : np.ndarray
        The array containing the data to be processed.
    mask : np.ndarray
        A boolean mask where True values indicate potential spikes.
    max_length_spike : int
        The maximum length of consecutive True values in the mask to be considered a spike.

    Returns:
    -------
    tuple[np.ndarray, int]
        A tuple containing the modified array with interpolated spike values and the total count of detected spikes.
    
    Notes:
    ------
    - If either the left or right neighbor is missing (i.e., the spike is at
      the boundary), the spike is not interpolated.
    - Only sequences of True values that are smaller than or equal to `max_length_spike` are considered spikes.
    """
    flag = False
    count_spike = 0
    for i in range(len(array)):
        if mask[i]:
            if flag == False: # first time of True value in the mask
                spike_indices = [i] # save the index of the first True
                flag = True # switch the flag
            else:
                spike_indices.append(i) # saves the indices of the true values consecutive to the first 
        else:
            if flag == True: # first time of False value after an isolated or a sequence of True values
                length_spike = len(spike_indices) # counts how many True values are in the closed sequence
                if length_spike <= max_length_spike: # the sequence is smaller than the maximum length => it's a spike
                    # controls if the left border of the sequence is the first element of the array
                    # or if the right border of the sequence is the last element of the array
                    # in that case the interpolation can be performed
                    if spike_indices[0] > 0 and spike_indices[-1] < len(array) - 1: 
                        left_value = array[spike_indices[0] - 1]
                        right_value = array[spike_indices[-1] + 1]
                        array[spike_indices] = linear_interp(left_value,
                                                             right_value,
                                                             length_spike)
                        count_spike = count_spike + 1 # increase the overall counter of spikes
                flag=False # reset the flag
                del spike_indices # clean the temporary variable
    return array, count_spike


def despiking_VM97(array_to_despike: np.ndarray,
                   c: float,
                   window_length: int,
                   max_consecutive_spikes: int,
                   max_iterations: int,
                   logger : Optional[logging.Logger]) -> np.ndarray:
    """
    Applies the despiking algorithm based on Vickers and Mahrt (1997) to remove spikes from a time series.

    This method identifies and replaces spikes in the input array by comparing values against a running
    mean and standard deviation computed over a moving window. Points lying beyond a threshold defined
    by `c` times the local standard deviation from the local mean are considered spikes. Spikes are 
    replaced using interpolation if their number is below `max_consecutive_spikes`.

    The threshold `c` is incrementally increased after each iteration.
    The process stops when no more spikes are detected or when `max_iterations` is reached.

    Parameters:
    ----------
    array_to_despike : np.ndarray
        The input 1D array containing the signal to be despiked.
    c : float
        Initial threshold multiplier for the standard deviation used to detect spikes.
    window_length : int
        Length of the moving window used to compute running statistics.
    max_consecutive_spikes : int
        Maximum number of consecutive spike points allowed for interpolation.
    max_iterations : int
        Maximum number of iterations to perform if spikes continue to be found.
    logger : Optional[logging.Logger], default=None
        A logger instance following the `logging.Logger` interface. If provided, the function will use it to
        log dialogues during the despiking procedure. If set to `None`, the function will operate
        silently without producing any log output.
    Returns:
    -------
    np.ndarray
        The despiked version of the input array.
    
    References:
    ----------
    Vickers, D., & Mahrt, L. (1997). Quality control and flux sampling problems for tower and aircraft data.
    Journal of Atmospheric and Oceanic Technology, 14(3), 512–526. https://doi.org/10.1175/1520-0426(1997)014<0512:QCAFSP>2.0.CO;2
    """

    iteration = 0
    c_increment = 0.1
    current_c = c
    array_despiked = array_to_despike.copy()
    count_spike = 1  # value > 0 to enter the cycle

    while count_spike != 0 and iteration <= max_iterations:
        running_mean, running_std = core.running_stats(array_despiked, window_length)

        if logger: logger.info(f"Iteration: {iteration}")

        upper_bound = running_mean + current_c * running_std
        lower_bound = running_mean - current_c * running_std

        beyond_bounds_mask = (array_despiked > upper_bound) | (array_despiked < lower_bound)

        array_despiked, count_spike = identify_interp_spikes(array_despiked,
                                                             beyond_bounds_mask,
                                                             max_consecutive_spikes)
        
        if logger: logger.info(f"Identified spikes: {count_spike}")

        current_c += c_increment  # increase the distance between the upper and lower bound
        iteration += 1

    return array_despiked


# def despiking_ROBUST(data : pd.DataFrame) -> pd.DataFrame:

#     data_despiked=data


#     return data_despiked


# def interp_nan(df, variable_list, test_mode, count_log, file_path, log_file, test_file):
#     # This function takes a df as input and returns a dataframe in the same order.
#     # Each column (except for the time column) is processed: if nans are detected, they are filled with interpolated values computed using
#     # linear interpolation of the first neighbors.

#     time_array=df.loc[:,'Time']
#     N=len(df)
#     for variable in variable_list:

#         array=df.loc[:,variable].to_numpy() #array su cui lavoro, proviene dal df di input
#         where_nan=np.isnan(array)
#         nan_initial=np.sum(where_nan)
#         nan_filled_count=0
#         dum=False

#         for i in range(N):
#             if where_nan[i]:
#                 if dum==False:
#                     spike_index=[i] #mi salvo le posizioni delle spike, la lunghezza di questo array definisce quante spike si susseguono
#                     dum=True
#                 else:
#                     spike_index.append(i)
#             else:
#                 if dum==True:
#                     #interpolazione lineare
#                     if (spike_index[0]==0) or (spike_index[-1]==(N-1)): #lascia invariato perchè non puoi interpolare
#                         dum=False
#                         del spike_index
#                     else:
#                         M=len(spike_index)
#                         y_0=array[spike_index[0]-1] #valore primo vicino a sinistra
#                         y_1=array[spike_index[-1]+1] #valore primo vicino a destra
#                         diff=y_1-y_0
#                         x=np.arange(1, M+1) #(1,2,3)
#                         array[spike_index]=y_0+x*diff/(M+1)                  
#                         dum=False #azzero il flag del contatore
#                         nan_filled_count=nan_filled_count+M
#                         del spike_index
#         df[variable]=array
#         del array, nan_filled_count, nan_initial, nan_final

#     return df

