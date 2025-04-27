import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional
# import src.core as core
import core

## controllo se possiede un numero di entrate pari alla frequenza* delta_t compreso tra l'inzio e la fine del 
def fill_missing_timestamps(data: pd.DataFrame, freq: float) -> pd.DataFrame:
    """
    Ensures a DataFrame with a datetime index includes all timestamps between 
    the first and last entry, based on the given frequency. Missing timestamps 
    are added with NaN values for all columns.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with a datetime index.
    freq : float
        Sampling frequency in Hertz (Hz).

    Returns
    -------
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
    array: np.ndarray,
    threshold: float,
    ) -> Tuple[np.ndarray, int]:
    """
    Replaces all values in the input array that exceed a given absolute threshold with NaN.

    Parameters
    ----------
    array : np.ndarray
        A NumPy array of numerical values to be cleaned.
    threshold : float
        Absolute threshold. All values with absolute magnitude greater than this will be set to NaN.

    Returns
    -------
    array_clean : np.ndarray
        A copy of the input array with values exceeding the threshold replaced by NaN.
    count_beyond : int
        The number of elements that were beyond the threshold and replaced.

    """
    array_clean = array.copy()
    where_beyond = np.abs(array_clean) > threshold
    count_beyond = np.count_nonzero(where_beyond)
    array_clean[where_beyond] = np.nan

    return array_clean, count_beyond

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

    Parameters
    ----------
    array : np.ndarray
        The array containing the data to be processed.
    mask : np.ndarray
        A boolean mask where True values indicate potential spikes.
    max_length_spike : int
        The maximum length of consecutive True values in the mask to be considered a spike.

    Returns
    -------
    tuple[np.ndarray, int]
        A tuple containing the modified array with interpolated spike values and the total count of detected spikes.
    
    Notes
    -----
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
                   logger : Optional[logging.Logger] = None) -> np.ndarray:
    """
    Applies the despiking algorithm based on Vickers and Mahrt (1997) to remove spikes from a time series.

    This method identifies and replaces spikes in the input array by comparing values against a running
    mean and standard deviation computed over a moving window. Points lying beyond a threshold defined
    by `c` times the local standard deviation from the local mean are considered spikes. Spikes are 
    replaced using interpolation if their number is below `max_consecutive_spikes`.

    The threshold `c` is incrementally increased after each iteration.
    The process stops when no more spikes are detected or when `max_iterations` is reached.

    Parameters
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
    
    Returns
    -------
    np.ndarray
        The despiked version of the input array.
    
    References
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
        running_mean, running_std = core.running_stats(array_despiked, 
                                                       window_length)

        upper_bound = running_mean + current_c * running_std
        lower_bound = running_mean - current_c * running_std

        beyond_bounds_mask = (array_despiked > upper_bound) | (array_despiked < lower_bound)

        array_despiked, count_spike = identify_interp_spikes(array_despiked,
                                                             beyond_bounds_mask,
                                                             max_consecutive_spikes)
        
        if logger: logger.info(
            f"""
            Iteration: {iteration}, identified and removed spikes: {count_spike}
            """)

        current_c += c_increment  # increase the distance between the upper and lower bound
        iteration += 1

    return array_despiked


def despiking_robust(array_to_despike: np.ndarray,
                     c: float,
                     window_length: int) -> np.ndarray:
    """
    Applies a non-iterative despiking algorithm using robust statistics to remove spikes from a time series.

    This method detects spikes by comparing each value in the input array against a local running median 
    and a robust estimate of the local variability, computed over a moving window. A point is classified 
    as a spike if it lies outside a dynamic threshold defined by `c` times the robust standard deviation 
    added to and subtracted from the running median. The robust standard deviation is defined as half the
    inter-percentile range between the 84th and 16th percentiles within the moving window:
    
    robust_std = (P84 - P16) / 2

    Detected spikes are replaced with the corresponding value of the running median. 
    This procedure is applied in a single pass and does not perform iterative refinement.

    Parameters
    ----------
    array_to_despike : np.ndarray
        The input 1D array containing the signal to be despiked.
    c : float
        Threshold multiplier for the robust standard deviation used to detect spikes.
    window_length : int
        Length of the moving window used to compute the running median and robust statistics.

    Returns
    -------
    np.ndarray
        The despiked version of the input array, with spikes replaced by the running median.
    int
        The total number of spikes detected and replaced.
    """
    array_despiked = array_to_despike.copy()

    running_median, running_std_robust = core.running_stats_robust(array_despiked, 
                                                                   window_length)
    delta = np.maximum(c * running_std_robust, 0.5)
    upper_bound = running_median + delta
    lower_bound = running_median - delta

    beyond_bounds_mask = (array_despiked > upper_bound) | (array_despiked < lower_bound)
    
    count_spike = np.sum(beyond_bounds_mask)

    array_despiked[beyond_bounds_mask] = running_median[beyond_bounds_mask] # replaces values with the running median where beyond computed bounds

    return array_despiked, count_spike


import numpy as np

def interp_nan(array: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Interpolates NaN values in the input array using linear interpolation.
    NaNs are replaced with values computed by the `linear_interp` function,
    using the closest non-NaN values to the left and right as reference points.
    If NaNs are at the edges of the array (i.e., without valid neighbors on both sides),
    they are left unchanged.

    Parameters
    ----------
    array : np.ndarray
        The input array containing NaN values to interpolate.

    Returns
    -------
    np.ndarray
        A copy of the input array with NaN values replaced by interpolated values,
        where possible.
    int
        Number of NaN values that were successfully interpolated.
    """
    array_interp = array.copy()
    isnan = np.isnan(array_interp)
    count_interp = 0

    i = 0
    while i < len(array_interp):
        if isnan[i]:
            start = i
            while i < len(array_interp) and isnan[i]:
                i += 1
            end = i

            if start == 0 or end == len(array_interp):
                continue  # cannot interpolate at the edges

            left_value = array_interp[start - 1]
            right_value = array_interp[end]
            length = end - start

            interpolated = linear_interp(left_value,
                                         right_value,
                                         length)
            
            array_interp[start:end] = interpolated

            count_interp += length
        else:
            i += 1

    return array_interp, count_interp


def rotation_to_LEC_reference(wind : np.ndarray,
                              azimuth : float,
                              model : str) -> np.ndarray: 
    """
    Rotate the wind vector from the anemometer reference system 
    to the Local Earth Coordinate system (LEC), given the orientation `azimuth`
    of the anemometer head with respect to the North.

    Parameters
    ----------
    wind : np.ndarray
        A 3xN array (shape (3, N)), where each column is a wind vector at a different time instant.
        The three rows correspond to the three velocity components.
    azimuth : float
        The azimuth angle in degrees measured clockwise from the North, describing the orientation
        of the anemometer head with respect to the North-
    model : str
        The anemometer model used for the measurement. 
        Only two models are supported:
            - "RM_YOUNG_81000"
            - "CAMPBELL_CSAT3"

    Returns
    -------
    np.ndarray
        A 3xN array (shape (3, N)) of wind vectors rotated into the LEC reference frame,
        with the y-axis aligned to the geographic North.
    
    Raises
    ------
    ValueError
        If the azimuth is outside the range [0, 360].
    ValueError
        If the anemometer model is not recognized (i.e., not "RM_YOUNG_81000" or "CAMPBELL_CSAT3").
    
    Notes
    -----
    The function applies two sequential rotations:
    1. A model-dependent transformation that maintains the Cartesian reference frame while 
    ensuring that the u and v wind components are positive when aligned with the x- and y-axes, respectively.
    2. A rotation aligning the y-axis to the North, according to the specified azimuth.
    """
    
    # ROTATION from the intrinsic reference system of the instrument
    # to a standard reference frame with:
    # u positive if concordant with x-axis
    # v positive if concordant with y-axis
    # with the y-axis oriented in the direction 
    # defined by the azimuth angle with respect to North (positive clockwise)
    # the rotation matrix depends on the anemometer model
    
    if not (0 <= azimuth <= 360):
        raise ValueError( "azimuth is outside the range [0,360].")
    
    rot_model = np.zeros((3,3))
    if model == "RM_YOUNG_81000":
        rot_model[0,0] = -1
        rot_model[1,1] = -1
        rot_model[2,2] =  1
    elif model == "CAMPBELL_CSAT3":
        rot_model[0,1] =  1
        rot_model[1,0] = -1
        rot_model[2,2] =  1
    else:
        raise ValueError(f"Unknown model: {model}")

    # ROTATION to LEC system with y-axis oriented to North
    azimuth = np.deg2rad(azimuth) # degree to radians conversion
    rot_azimuth = np.zeros((3,3))
    rot_azimuth[0, 0] =  np.cos(azimuth) # input have to be angles in radians
    rot_azimuth[0, 1] =  np.sin(azimuth)
    rot_azimuth[1, 0] = -np.sin(azimuth)
    rot_azimuth[1, 1] =  np.cos(azimuth)
    rot_azimuth[2, 2] =  1

    rot_total = np.matmul(rot_azimuth, rot_model)
    wind_rotated = np.matmul(rot_total, wind)
    
    return wind_rotated

def rotation_to_streamline_reference(wind : np.ndarray, # model indipendent
                                    wind_averaged : np.ndarray) -> np.ndarray:
    # controllo sulla dimensione di wind e wind averaged
    wind_rotated = np.full(wind.shape, np.nan)
    return wind_rotated

def wind_dir_LEC_reference(horizontal_wind : np.ndarray) -> np.ndarray:
    # controllo sulla shape di horizontal wind
    N = horizontal_wind.shape[1]
    wind_direction = np.full(N, np.nan)
    return wind_direction

def wind_dir_modeldependent_reference(wind : np.ndarray,
                                      azimuth : float,
                                      model : str) -> np.ndarray:
    # controllo sulla shape di wind averaged, valore di azimuth e modelli possibili
    N = wind.shape[1]
    wind_direction = np.full(N, np.nan)
    return wind_direction # model dependent computation