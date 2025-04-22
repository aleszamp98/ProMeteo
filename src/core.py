import pandas as pd
import numpy as np
import os
import configparser
from typing import Tuple
from numpy.lib.stride_tricks import sliding_window_view
import warnings

def load_config(path : str) -> dict:
    """
    Loads parameters from a config.txt file and returns them as a dictionary.

    Parameters
    ----------
    path : str or Path
        Path to the config.txt file.

    Returns
    -------
    dict
        Dictionary containing the parameters for main.py .

    Raises
    ------
    FileNotFoundError
        If the file does not exist or cannot be read.
    configparser.NoSectionError
        If a required section is missing.
    configparser.NoOptionError
        If a required option is missing.
    ValueError
        If a parameter cannot be converted to the expected type.
    """
    config = configparser.ConfigParser()
    files_read = config.read(path)

    try:
        rawdata_path = config.get('general', 'rawdata_path')
        dir_out = config.get('general', 'dir_out')
        sampling_freq_str = config.get('general', 'sampling_freq')

        horizontal_threshold_str = config.get('remove_beyond_threshold', 'horizontal_threshold')
        vertical_threshold_str = config.get('remove_beyond_threshold', 'vertical_threshold')
        temperature_threshold_str = config.get('remove_beyond_threshold', 'temperature_threshold')

        despiking_mode = config.get('despiking', 'despiking_mode')
        window_length_despiking = config.get('despiking', 'window_length_despiking')
        max_n_consecutive_values = config.get('despiking', 'max_n_consecutive_values')
        max_iterations = config.get('despiking', 'max_iterations')
        c_H = config.get('despiking', 'c_H')
        c_V = config.get('despiking', 'c_V')
        c_T = config.get('despiking', 'c_T')

    except configparser.NoSectionError as e:
        raise configparser.NoSectionError(e.section) from e
    except configparser.NoOptionError as e:
        raise configparser.NoOptionError(e.option, e.section) from e
    
    # control over passed sampling frequency
    try:
        sampling_freq = int(sampling_freq_str)
    except ValueError as e:
        raise ValueError(f"'sampling_freq' must be an integer, got '{sampling_freq_str}' instead.") from e
    
    # control over passed thresholds
    try:
        horizontal_threshold = float(horizontal_threshold_str)
        vertical_threshold = float(vertical_threshold_str)
        temperature_threshold = float(temperature_threshold_str)
    except ValueError as e:
        raise ValueError(
            "Threshold values must be numbers.\n"
            f"Got: horizontal='{horizontal_threshold_str}', vertical='{vertical_threshold_str}', temperature='{temperature_threshold_str}'"
        ) from e
    
    # control over despiking_mode
    allowed_modes = ['VM97', 'ROBUST']
    if despiking_mode not in allowed_modes:
        raise ValueError(f"Invalid value for 'despiking_mode': '{despiking_mode}'. "
                         f"Allowed values are: {allowed_modes}")
    
    # control over passed inputs for despiking procedure
    try:
        window_length_despiking = int(window_length_despiking)
        max_n_consecutive_values = int(max_n_consecutive_values)
        max_iterations = int(max_iterations)
    except ValueError as e:
        raise ValueError(
            "window_length_despiking, max_n_consecutive_values and max_iterations must be integers.\n"
            f"Got: window_length_despiking='{window_length_despiking}', max_n_consecutive_values='{max_n_consecutive_values}' and max_iterations='{max_iterations}'"
        ) from e
    try:
        c_H = float(c_H)
        c_V = float(c_V)
        c_T = float(c_T)
    except ValueError as e:
        raise ValueError(
            "c_H, c_V and c_T must be numbers.\n"
            f"Got: c_H='{c_H}', c_V='{c_V}', c_T='{c_T}'"
        ) from e

    params = {
        'rawdata_path': rawdata_path,
        'dir_out': dir_out,
        'sampling_freq': sampling_freq,
        'horizontal_threshold': horizontal_threshold,
        'vertical_threshold': vertical_threshold,
        'temperature_threshold': temperature_threshold,
        'despiking_mode': despiking_mode,
        'window_length_despiking' : window_length_despiking,
        'max_n_consecutive_values' : max_n_consecutive_values,
        'max_iterations' : max_iterations,
        'c_H' : c_H,
        'c_V' : c_V,
        'c_T' : c_T,
    }


    return params



def import_data(path : str) -> pd.DataFrame:

    """
    Imports a CSV file containing data collected from sonic anemometer.

    The file must contain the columns in the order: ["Time", "u", "v", "w", "T_s"],
    where "Time" is a column with valid timestamps and the other columns contain float values.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file to be read.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the timestamp as the index (from the "Time" column) and columns ["u", "v", "w", "T_s"].

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the specified path.
    ValueError
        If the columns in the file do not match the expected order, if any timestamp is invalid,
        or if the "Time" column contains values that cannot be converted to datetime.
    """

    expected_columns = ["Time", "u", "v", "w", "T_s"]

    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} doesn't exist.")
    
    data = pd.read_csv(path, sep=",", header=0)

    if list(data.columns) != expected_columns:
        raise ValueError(f"Expected columns: {expected_columns}, found: {list(data.columns)}")
    
    # read timestamps as datetime
    data["Time"] = pd.to_datetime(data["Time"], errors="coerce")
    if data["Time"].isna().any():
        raise ValueError("'Time' column contains non-valid values.")
    data = data.set_index("Time") # "Time" col set as index

    return data

def min_to_points(minutes: int,
                  sampling_freq: int
                  ) -> int:
    """
    Computes the number of data points contained in a signal of known sampling frequency
    given the time lenght in minutes.

    Parameters
    ----------
    minutes : int
        Duration of the signal in minutes.
    sampling_freq : int
        Sampling frequency in Hz (samples per second).

    Returns
    -------
    int
        Total number of data points in the signal for the given duration.
    """
    n_points = sampling_freq * minutes * 60
    return n_points

def running_stats(array: np.ndarray, window_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the running (moving) mean and standard deviation of a 1D array using a sliding window.

    This function uses a centered sliding window of fixed length to calculate
    the mean and standard deviation at each point in the input array. The array
    is padded at the edges to allow computation at the boundaries.

    NaN values within the window are ignored in the calculations.

    Parameters
    ----------
    array : np.ndarray
        Input 1D array of numerical values.
    window_length : int
        Length of the sliding window. Must be a positive odd integer less than or equal to the array length.

    Returns
    -------
    running_mean : np.ndarray
        Array of the same length as the input, containing the running mean.
    running_std : np.ndarray
        Array of the same length as the input, containing the running standard deviation.

    Raises
    ------
    ValueError
        If `window_length` is not a positive integer.
        If `window_length` is greater than the length of the input array.

    Warns
    -----
    UserWarning
        If `window_length` is even, a warning is issued that using an even-length window may result in asymmetric behavior.

    Notes
    -----
    The function pads the input array using edge values to maintain the original length
    in the output. If the input contains NaNs, they are ignored in the mean and std computation
    using `np.nanmean` and `np.nanstd`.
    """
    if not isinstance(window_length, int) or window_length <= 0:
        raise ValueError("window_length must be a positive integer.")

    if window_length > len(array):
        raise ValueError("window_length must not be greater than the length of the array.")

    if window_length % 2 == 0:
        warnings.warn(
            "window_length is even; using an even-length window may result in asymmetric behavior.",
            UserWarning
        )

    half_window = window_length // 2

    padded = np.pad(array, (half_window, half_window), mode='edge') # pad the array at the beginning and end, repeating the values at the edges
    windows = sliding_window_view(padded, window_shape=window_length) # create sliding windows

    running_mean = np.nanmean(windows, axis=1)
    running_std = np.nanstd(windows, axis=1)

    return running_mean, running_std