import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, Union
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import core as core

def fill_missing_timestamps(data : pd.DataFrame, 
                            freq : float
                            ) -> pd.DataFrame:
    """
    Returns the input DataFrame data with a complete datetime index: 
    all timestamps between the first and last entry are included 
    based on the specified frequency, and missing timestamps are filled 
    with rows containing NaN values.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with a datetime index.
    freq : float
        Sampling frequency in Hertz (Hz).

    Returns
    -------
    complete_data : pd.DataFrame
        DataFrame reindexed to include all expected timestamps.
    
    Raises
    ------
    ValueError
        If `freq` is negative.
    """
    # --- Input validation ---
    if freq <= 0:
        raise ValueError("Frequency (Hz) must be a positive number.")
    
    dt_s=1/freq
    freq = pd.to_timedelta(dt_s, unit='s') # pandas-compatible frequency string
    data = data.sort_index()    
    # Create the complete range of timestamps
    full_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq=freq) 
    # Reindex the DataFrame to include all timestamps
    complete_data = data.reindex(full_index)
    
    return complete_data
    
def remove_beyond_threshold(array : np.ndarray,
                            threshold : float
                            ) -> Tuple[np.ndarray, int]:
    """
    Replaces all values in the input array that exceed a given absolute `threshold` with NaN.

    Parameters
    ----------
    array : np.ndarray
        The 1D array of numerical values to be cleaned.
    threshold : float
        Absolute threshold. All values with absolute magnitude greater than this will be set to NaN.

    Returns
    -------
    array_clean : np.ndarray
        A copy of the input array with values exceeding the threshold replaced by NaN.
    count_beyond : int
        The number of elements that were beyond the threshold and replaced.
    
    Raises
    ------
    ValueError
        If `threshold` is negative.
    """
    # --- Input validation ---
    if threshold < 0:
        raise ValueError(f" Threshold must be positive.")
    
    array_clean = array.copy()
    where_beyond = np.abs(array_clean) > threshold
    count_beyond = np.count_nonzero(where_beyond)
    array_clean[where_beyond] = np.nan

    return array_clean, count_beyond

def linear_interp(left_value : float,
                  right_value : float,
                  length : int
                  ) -> np.ndarray:
    """
    Performs linear interpolation between `left_value` and `right_value`, 
    returning a NumPy array of length `length` whose elements represent 
    evenly spaced values between the two endpoints.
    
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
    interpolated_values : np.ndarray
        An array containing the interpolated values.
    
    Raises
    ------
    ValueError
        If `length` is not a positive integer.

    """
    # --- Input validation ---
    if not isinstance(length, int) or length <= 0:
        raise ValueError("`length` must be a positive integer.")

    # Create an array of relative indices
    x = np.arange(1, length + 1)
    
    # Compute the interpolation
    interpolated_values = left_value + (x / (length + 1)) * (right_value - left_value)
    
    return interpolated_values

def identify_interp_spikes(array : np.ndarray,
                           mask : np.ndarray,
                           max_length_spike : int
                           ) -> tuple[np.ndarray, int]:
    """
    Identifies and interpolates spikes in the provided `array` based on the given mask.
    
    A spike is defined as a sequence of consecutive True values in the mask that is shorter than or equal to
    the specified maximum length (`max_length_spike`). The function applies linear interpolation to replace 
    the spike values with interpolated ones by calling `core.linear_interp()`. If the spike is at the boundary 
    of the array (either at the start or at the end), interpolation is not performed.

    Parameters
    ----------
    array : np.ndarray
        The 1D array containing the data to be processed.
    mask : np.ndarray
        A boolean mask where True values indicate potential spikes.
    max_length_spike : int
        The maximum length of consecutive True values in the mask to be considered a spike.

    Returns
    -------
    array : np.ndarray
        The modified array with interpolated spike values
    count_spike : int
        The total count of detected spikes.
    
    Raises
    ------
    ValueError
        If `array` and `mask` do not have the same length.
    ValueError
        If `mask` is not a boolean array.
    ValueError
        If `max_length_spike` is not a positive integer.

    Notes
    -----
    - If either the left or right neighbor is missing (i.e., the spike is at
      the boundary), the spike is not interpolated.
    - Only sequences of True values that are smaller than or equal to `max_length_spike` are considered spikes.
    """

     # --- Input validation ---
    if len(array) != len(mask):
        raise ValueError("`array` and `mask` must have the same length.")
    if mask.dtype != bool:
        raise ValueError("`mask` must be a boolean array.")
    if not isinstance(max_length_spike, int) or max_length_spike <= 0:
        raise ValueError("`max_length_spike` must be a positive integer.")

    # --- Spike detection and interpolation ---
    flag = False  # Flag to track the beginning of a spike sequence
    count_spike = 0  # Counter to keep track of the total number of spikes detected

    # Loop through the array to check each element
    for i in range(len(array)):
        # Check if the mask indicates a True value at index i
        if mask[i]:
            # If flag is False, it's the first time encountering a True value in the mask
            if flag == False:
                spike_indices = [i]  # Save the index of the first True value
                flag = True  # Set the flag to True indicating a spike sequence has started
            else:
                # If the flag is True, append the index of consecutive True values
                spike_indices.append(i)
        else:
            # If the current mask value is False and a spike sequence was detected (flag is True)
            if flag == True:
                # Count the length of the detected spike sequence
                length_spike = len(spike_indices)
                # If the sequence is shorter than or equal to the max allowed spike length, treat it as a spike
                if length_spike <= max_length_spike:
                    # Check if interpolation is possible (the spike is not at the border of the array)
                    if spike_indices[0] > 0 and spike_indices[-1] < len(array) - 1:
                        # Retrieve the neighboring values for interpolation
                        left_value = array[spike_indices[0] - 1]
                        right_value = array[spike_indices[-1] + 1]
                        # Perform linear interpolation to smooth the spike
                        array[spike_indices] = linear_interp(left_value, right_value, length_spike)
                        # Increment the spike count
                        count_spike += 1
                # Reset the flag and clean up the temporary variables
                flag = False
                del spike_indices  # Delete the spike indices list to free memory

    return array, count_spike



def despiking_VM97(array_to_despike : np.ndarray,
                   c : float,
                   window_length : int,
                   max_consecutive_spikes : int,
                   max_iterations : int,
                   logger : Optional[logging.Logger] = None
                   ) -> np.ndarray:
    """
    Applies the despiking algorithm based on Vickers and Mahrt (1997) to remove spikes from a time series.

    This method identifies and replaces spikes in the input array by comparing values against a running
    mean and standard deviation computed over a moving window. Points lying beyond a threshold defined
    by `c` times the local standard deviation from the local mean are considered spikes. Spikes are 
    replaced using interpolation if their number is below `max_consecutive_spikes`.

    The `c` factor is incrementally increased after each iteration.
    The process stops when no more spikes are detected or when `max_iterations` is reached.

    The function calls `pre_processing.identify_inter_spikes()` and `core.running_stats()` functions.

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
    array_despiked : np.ndarray
        The despiked version of the input array.
    
    Raises
    ------
    ValueError
        If `c` is not a positive number.
    ValueError
        If `window_length` is not a positive integer.
    ValueError
        If `max_consecutive_spikes` is not a positive integer.
    ValueError
        If `max_iterations` is not a positive integer.
    ValueError
        If `logger` is not a logging.Logger instance or None.
    
    References
    ----------
    Vickers, D., & Mahrt, L. (1997). Quality control and flux sampling problems for tower and aircraft data.
    Journal of Atmospheric and Oceanic Technology, 14(3), 512–526. https://doi.org/10.1175/1520-0426(1997)014<0512:QCAFSP>2.0.CO;2
    """
    
    # --- Input validation ---
    if not (isinstance(c, (int, float)) and c > 0):
        raise ValueError("`c` must be a positive number.")
    if not isinstance(window_length, int) or window_length <= 0:
        raise ValueError("`window_length` must be a positive integer.")
    if not isinstance(max_consecutive_spikes, int) or max_consecutive_spikes <= 0:
        raise ValueError("`max_consecutive_spikes` must be a positive integer.")
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("`max_iterations` must be a positive integer.")
    
    # --- Despiking process ---
    iteration = 0  # Counter for the number of iterations in the despiking process
    c_increment = 0.1  # Increment value for adjusting the bounds in each iteration
    current_c = c  # Initial multiplier for the standard deviation (controls the bounds)
    array_despiked = array_to_despike.copy()  # Copy of the original array to be despiked
    count_spike = 1  # Initial value greater than 0 to start the despiking cycle

    # The loop continues until no more spikes are detected or the maximum number of iterations is reached
    max_iterations = max_iterations - 1  # Adjust for zero-based counting
    while count_spike != 0 and iteration <= max_iterations:
        # Calculate the running mean and standard deviation for the current array
        running_mean, running_std = core.running_stats(array_despiked, window_length)

        # Define the upper and lower bounds based on the running statistics and current multiplier (current_c)
        upper_bound = running_mean + current_c * running_std
        lower_bound = running_mean - current_c * running_std

        # Create a mask to detect values beyond the defined bounds
        beyond_bounds_mask = (array_despiked > upper_bound) | (array_despiked < lower_bound)

        # Identify and interpolate values that are beyond the bounds and respect max_length criterion (spikes)
        array_despiked, count_spike = identify_interp_spikes(array_despiked, beyond_bounds_mask, max_consecutive_spikes)
        
        # Log the progress if the logger is enabled
        if logger:
            logger.info(f"""
                Iteration: {iteration}, identified and removed spikes: {count_spike}
            """)

        # Increase the multiplier for the next iteration to make the bounds wider
        current_c += c_increment
        iteration += 1  # Increment the iteration counter

    return array_despiked


def despiking_robust(array_to_despike : np.ndarray,
                     c : float,
                     window_length : int
                     ) -> Tuple[np.ndarray, int]:
    """
    Applies a non-iterative despiking algorithm using robust statistics to remove spikes from a time series.

    This method detects spikes by comparing each value in the input array against a local running median 
    and a robust estimate of the local variability, computed over a moving window. A point is classified 
    as a spike if it lies outside a dynamic threshold defined by `c` times the robust standard deviation 
    added to and subtracted from the running median. The robust standard deviation is defined as half the
    inter-percentile range between the 84th and 16th percentiles within the moving window.

    Detected spikes are replaced with the corresponding value of the running median. 
    This procedure is applied in a single pass and does not perform iterative refinement.

    The function calls `core.running_stats_robust()` function.

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
    array_despiked : np.ndarray
        The despiked version of the input array, with spikes replaced by the running median.
    count_spike : int
        The total number of spikes detected and replaced.

    Raises
    ------
    ValueError
        If `c` is not positive.
    ValueError
        If `window_length` is not a positive integer.
    """
    # --- Input validation ---
    if not isinstance(c, (int, float)) or c <= 0:
        raise ValueError("Parameter `c` must be a positive number.")
    if not isinstance(window_length, int) or window_length <= 0:
        raise ValueError("Parameter `window_length` must be a positive integer.")

    array_despiked = array_to_despike.copy()

    # Calculate the running median and robust standard deviation for the array
    running_median, running_std_robust = core.running_stats_robust(array_despiked, window_length)

    # Compute the delta (threshold for spike detection) as the maximum between c times the robust standard deviation and 0.5
    delta = np.maximum(c * running_std_robust, 0.5)

    # Define the upper and lower bounds based on the running median and the computed delta
    upper_bound = running_median + delta
    lower_bound = running_median - delta

    # Create a mask to detect values that lie outside the defined bounds (spikes)
    beyond_bounds_mask = (array_despiked > upper_bound) | (array_despiked < lower_bound)

    # Count the number of spikes (values that are beyond the bounds)
    count_spike = np.sum(beyond_bounds_mask)

    # Replace the values that are outside the bounds with the corresponding values from the running median
    array_despiked[beyond_bounds_mask] = running_median[beyond_bounds_mask]

    return array_despiked, count_spike


def interp_nan(array : np.ndarray
               ) -> Tuple[np.ndarray, int]:
    """
    Interpolates NaN values in the input array using linear interpolation.
    NaNs are replaced with values computed by `core.linear_interp()` function,
    using the closest non-NaN values to the left and right as reference points.
    If NaNs are at the edges of the array (i.e., without valid neighbors on both sides),
    they are left unchanged.

    Parameters
    ----------
    array : np.ndarray
        The input array containing NaN values to interpolate.

    Returns
    -------
    array_interp : np.ndarray
        A copy of the input array with NaN values replaced by interpolated values, where possible.
    count_interp : int
        Number of NaN values that were successfully interpolated.
    """
    array_interp = array.copy()
    # Identify where NaN values are present in the array
    isnan = np.isnan(array_interp)

    # Initialize a counter for the number of interpolated values
    count_interp = 0

    # Initialize the index to iterate through the array
    i = 0
    while i < len(array_interp):
        # If a NaN value is found, start the interpolation process
        if isnan[i]:
            start = i  # Store the index of the first NaN value in the sequence
            # Continue moving the index while NaN values are encountered
            while i < len(array_interp) and isnan[i]:
                i += 1
            end = i  # Store the index where the NaN sequence ends

            # Skip interpolation if the NaN sequence is at the edges of the array
            if start == 0 or end == len(array_interp):
                continue

            # Get the surrounding values for interpolation (left and right of the NaN sequence)
            left_value = array_interp[start - 1]
            right_value = array_interp[end]

            # Calculate the length of the NaN sequence
            length = end - start

            # Perform linear interpolation for the NaN values
            interpolated = linear_interp(left_value, right_value, length)

            # Replace the NaN values with the interpolated values
            array_interp[start:end] = interpolated

            # Update the counter for the number of interpolated values
            count_interp += length
        else:
            # Move to the next index if the current value is not NaN
            i += 1

    return array_interp, count_interp