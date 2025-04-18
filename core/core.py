import numpy as np
import pandas as pd
import os


def import_data(path):

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