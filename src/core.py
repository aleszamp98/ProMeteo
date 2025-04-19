import pandas as pd
import os
import configparser


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
    except configparser.NoSectionError as e:
        raise configparser.NoSectionError(e.section) from e
    except configparser.NoOptionError as e:
        raise configparser.NoOptionError(e.option, e.section) from e

    try:
        sampling_freq = int(sampling_freq_str)
    except ValueError as e:
        raise ValueError(f"'sampling_freq' must be an integer, got '{sampling_freq_str}' instead.") from e
    
    try:
        horizontal_threshold = float(horizontal_threshold_str)
        vertical_threshold = float(vertical_threshold_str)
        temperature_threshold = float(temperature_threshold_str)
    except ValueError as e:
        raise ValueError(
            "Threshold values must be float-compatible.\n"
            f"Got: horizontal='{horizontal_threshold_str}', vertical='{vertical_threshold_str}', temperature='{temperature_threshold_str}'"
        ) from e

    params = {
        'rawdata_path': rawdata_path,
        'dir_out': dir_out,
        'sampling_freq': sampling_freq,
        'horizontal_threshold': horizontal_threshold,
        'vertical_threshold': vertical_threshold,
        'temperature_threshold': temperature_threshold,
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