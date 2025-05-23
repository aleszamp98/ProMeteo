import pandas as pd
import numpy as np
import os
import configparser
from typing import Tuple
import warnings

def load_config(path : str) -> dict:
    """
    Load and validate parameters from a configuration file.

    This function reads a configuration file in INI format (e.g., config.txt),
    extracts and validates the necessary parameters, and returns them in a dictionary.

    Parameters
    ----------
    path : str
        Path to the configuration file (e.g., 'config.txt').

    Returns
    -------
    dict
        Dictionary containing validated parameters required by `main.py`.

    Raises
    ------
    FileNotFoundError
        If the configuration file cannot be found or read.
    configparser.NoSectionError
        If a required section is missing from the configuration.
    configparser.NoOptionError
        If a required option is missing from a section.
    ValueError
        If a parameter has an invalid value or cannot be converted to its expected type.
    """

    config = configparser.ConfigParser()
    files_read = config.read(path)
    if not files_read:
        raise FileNotFoundError(f"Configuration file '{path}' not found or not readable.")

    try:
        # General
        rawdata_path = config.get('general', 'rawdata_path')
        dir_out = config.get('general', 'dir_out')
        sampling_freq = config.getfloat('general', 'sampling_freq')
        model = config.get('general', 'model')

        # Thresholds
        horizontal_threshold = config.getfloat('remove_beyond_threshold', 'horizontal_threshold')
        vertical_threshold = config.getfloat('remove_beyond_threshold', 'vertical_threshold')
        temperature_threshold = config.getfloat('remove_beyond_threshold', 'temperature_threshold')

        # Despiking
        despiking_method = config.get('despiking', 'despiking_method')
        window_length_despiking = config.getfloat('despiking', 'window_length_despiking')
        max_length_spike = config.getint('despiking', 'max_length_spike')
        max_iterations = config.getint('despiking', 'max_iterations')
        c_H = config.getfloat('despiking', 'c_H')
        c_V = config.getfloat('despiking', 'c_V')
        c_T = config.getfloat('despiking', 'c_T')
        c_robust = config.getfloat('despiking', 'c_robust')

        # Averaging
        window_length_averaging = config.getfloat('averaging', 'window_length_averaging')

        # Rotation and wind direction
        reference_frame = config.get('rotation', 'reference_frame')
        azimuth = config.getfloat('rotation', 'azimuth')
        wind_dir_threshold = config.getfloat('rotation', 'wind_dir_threshold')

    except configparser.NoSectionError as e:
        raise configparser.NoSectionError(e.section) from e
    except configparser.NoOptionError as e:
        raise configparser.NoOptionError(e.option, e.section) from e
    except ValueError as e:
        raise ValueError(f"Invalid type in config file: {e}") from e

    # Validations
    allowed_models = ['RM_YOUNG_81000', 'CAMPBELL_CSAT3']
    if model not in allowed_models:
        raise ValueError(f"Invalid input for 'model': '{model}'. Allowed values: {allowed_models}")

    allowed_methods = ['VM97', 'robust']
    if despiking_method not in allowed_methods:
        raise ValueError(f"Invalid 'despiking_method': '{despiking_method}'. Allowed values: {allowed_methods}")

    allowed_reference_frames = ['LEC', 'streamline']
    if reference_frame not in allowed_reference_frames:
        raise ValueError(f"Invalid 'reference_frame': '{reference_frame}'. Allowed values: {allowed_reference_frames}")

    if not (0 <= azimuth <= 360):
        raise ValueError(f"azimuth must be between 0 and 360. Got: {azimuth}")

    return {
        'rawdata_path': rawdata_path,
        'dir_out': dir_out,
        'sampling_freq': sampling_freq,
        'model': model,
        'horizontal_threshold': horizontal_threshold,
        'vertical_threshold': vertical_threshold,
        'temperature_threshold': temperature_threshold,
        'despiking_method': despiking_method,
        'window_length_despiking': window_length_despiking,
        'max_length_spike': max_length_spike,
        'max_iterations': max_iterations,
        'c_H': c_H,
        'c_V': c_V,
        'c_T': c_T,
        'c_robust': c_robust,
        'window_length_averaging': window_length_averaging,
        'reference_frame': reference_frame,
        'azimuth': azimuth,
        'wind_dir_threshold' : wind_dir_threshold, 
    }

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
    data : pd.DataFrame
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

def min_to_points(minutes : int,
                  sampling_freq : int
                  ) -> int:
    """
    Computes the number of data points contained in a signal of known sampling frequency
    given the time length in minutes, ensuring that the result is always an odd number,
    unless the result is 0, in which case it returns 0.

    Parameters
    ----------
    minutes : int
        Duration of the signal in minutes.
    sampling_freq : int
        Sampling frequency in Hz (samples per second).

    Returns
    -------
    n_points : int
        Total number of data points in the signal for the given duration, adjusted to be odd if necessary.
        Returns 0 if the result is zero.
    """
    n_points = sampling_freq * minutes * 60
    if n_points == 0:
        pass  # Keep n_points as 0
    elif n_points % 2 == 0:  # If n_points is even, add 1 to make it odd
        n_points += 1
    return int(n_points)

def running_stats(array: np.ndarray, 
                  window_length: int
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the running (moving) mean and standard deviation of a 1D array using a sliding window.

    This function uses a centered sliding window of fixed length to calculate
    the mean and standard deviation at each point in the input array. The array
    is padded at the edges to allow computation at the boundaries.

    Parameters
    ----------
    array : np.ndarray
        Input 1D array of numerical values.
    window_length : int
        Number of points within the sliding window. Must be a positive odd integer less than or equal to the array length.

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
    - The function pads the input array using constant values equals to the edge values of the array.
    - If the input contains NaNs, they are ignored in the mean and std computation using `np.nanmean` and `np.nanstd`.
    """
    # --- Input validation ---
    if not isinstance(window_length, int) or window_length <= 0:
        raise ValueError("window_length must be a positive integer.")
    if window_length > len(array):
        raise ValueError("window_length must not be greater than the length of the array.")
    if window_length % 2 == 0:
        warnings.warn(
            "window_length is even; using an even-length window may result in asymmetric behavior.",
            UserWarning)

    # --- Initialization ---
    N = len(array)
    half_window = window_length // 2

    # Pad the array to handle edge
    # The padding replicates the boundary values of the array
    padded_array = np.pad(array, (half_window, half_window), mode='edge')

    running_mean = np.full(N, np.nan)
    running_std = np.full(N, np.nan)

    # --- Running statistics calculation ---
    for i in range(N):
        # Define the window centered around index 'i'
        window = padded_array[i:i + window_length]
        
        # Calculate the running mean and standard deviation for the current window, ignoring NaN values
        running_mean[i] = np.nanmean(window)
        running_std[i] = np.nanstd(window)

    return running_mean, running_std

def running_stats_robust(array : np.ndarray, 
                  window_length : int
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the running (moving) median and robust standard deviation of a 1D array using a sliding window.

    This function uses a centered sliding window of fixed length to calculate
    the median and a robust estimate of the standard deviation at each point in the input array.
    The robust standard deviation is computed as half the difference between the 84th and 16th percentiles.
    The array is padded at the edges to allow computation at the boundaries.

    Parameters
    ----------
    array : np.ndarray
        Input 1D array of numerical values.
    window_length : int
        Number of points within the sliding window. Must be a positive odd integer less than or equal to the array length.

    Returns
    -------
    running_median : np.ndarray
        Array of the same length as the input, containing the running median.
    running_std_robust : np.ndarray
        Array of the same length as the input, containing the robust running standard deviation 
        computed as (84th percentile - 16th percentile) / 2.

    Raises
    ------
    ValueError
        If `window_length` is not a positive integer.
    ValueError
        If `window_length` is greater than the length of the input array.

    Warns
    -----
    UserWarning
        If `window_length` is even, a warning is issued that using an even-length window may result in asymmetric behavior.

    Notes
    -----
    - The function pads the input array using constant values equal to the edge values of the array.
    - If the input contains NaNs, they are ignored in the percentile and median computations
      using `np.nanpercentile` and `np.nanmedian`.
    """
    # --- Input validation ---
    if not isinstance(window_length, int) or window_length <= 0:
        raise ValueError("window_length must be a positive integer.")
    if window_length > len(array):
        raise ValueError("window_length must not be greater than the length of the array.")
    if window_length % 2 == 0:
        warnings.warn(
            "window_length is even; using an even-length window may result in asymmetric behavior.",
            UserWarning)

    # --- Initialization ---
    N = len(array)
    half_window = window_length // 2

    # Pad the array to handle edge
    # The padding replicates the boundary values of the array
    padded_array = np.pad(array, (half_window, half_window), mode='edge')

    running_median = np.full(N, np.nan)
    running_std_robust = np.full(N, np.nan)

    # --- Running statistics calculation ---
    for i in range(N):
        # Define the window centered around index 'i'
        window = padded_array[i:i + window_length]
        
        # Calculate the running median for the current window, ignoring NaN values
        running_median[i] = np.nanmedian(window)
        
        # Calculate the 84th and 16th percentiles of the window to compute the robust standard deviation
        p84 = np.nanpercentile(window, 84)
        p16 = np.nanpercentile(window, 16)
        
        # Robust standard deviation is defined as half the difference between the 84th and 16th percentiles
        running_std_robust[i] = 0.5 * (p84 - p16)

    return running_median, running_std_robust