import pytest
from numpy.testing import assert_array_equal, assert_almost_equal
import pandas as pd
import numpy as np
# import os 
import sys
import configparser
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
import core


##### testing core.load_config() #####

# def write_config_file(tmp_path, content: str):
#     config_file = tmp_path / "config.txt"
#     config_file.write_text(content)
#     return config_file

# def test_valid_config(tmp_path):
#     content = """
#     [general]
#     rawdata_path = ./data.csv
#     dir_out = ./output/
#     sampling_freq = 20
#     """
#     config_path = write_config_file(tmp_path, content)
#     params = core.load_config(config_path)
#     assert params['sampling_freq'] == 20
#     assert isinstance(params['rawdata_path'], str)
#     assert isinstance(params['dir_out'], str)

# def test_missing_section(tmp_path):
#     content = """
#     [wrong_section]
#     rawdata_path = ./data.csv
#     dir_out = ./output/
#     sampling_freq = 10
#     """
#     config_path = write_config_file(tmp_path, content)
#     with pytest.raises(configparser.NoSectionError):
#         core.load_config(config_path)

# def test_missing_option(tmp_path):
#     content = """
#     [general]
#     rawdata_path = ./data.csv
#     # dir_out missing
#     sampling_freq = 10
#     """
#     config_path = write_config_file(tmp_path, content)
#     with pytest.raises(configparser.NoOptionError):
#         core.load_config(config_path)

# def test_invalid_type(tmp_path):
#     content = """
#     [general]
#     rawdata_path = ./data.csv
#     dir_out = ./output/
#     sampling_freq = ten
#     """
#     config_path = write_config_file(tmp_path, content)
#     with pytest.raises(ValueError, match="sampling_freq.*integer"):
#         core.load_config(config_path)

# def test_config_read_empty(tmp_path):
#     config_path = tmp_path / "config_missing.txt"
#     config = configparser.ConfigParser()
#     read_files = config.read(config_path)
#     assert read_files == []  # Deve restituire una lista vuota


##### testing core.import_data() #####

def test_import_valid_file(tmp_path):
    file_path = tmp_path / "try.csv"
    file_content = (
        "Time,u,v,w,T_s\n"
        "2024-01-01 00:00:00,1.0,2.0,3.0,25.0\n"
        "2024-01-01 00:01:00,1.1,2.1,3.1,25.1\n"
    )
    file_path.write_text(file_content)
    
    df = core.import_data(file_path)
    assert isinstance(df, pd.DataFrame) # Test if the variable is a DataFrame
    assert list(df.columns) == ["u", "v", "w", "T_s"] # Test if the columns are the expected ones
    assert isinstance(df.index[0], pd.Timestamp) # Test if the index is in datetime format

def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        core.import_data("fail.csv") # Test a failing import

def test_invalid_columns(tmp_path):
    file_path = tmp_path / "bad_columns.csv"
    file_content = (
        "Timestamp,u,v,w,T_s\n"  # "Time" -> "Timestamp"
        "2024-01-01 00:00:00,1.0,2.0,3.0,25.0\n"
    )
    file_path.write_text(file_content)

    with pytest.raises(ValueError, match="Expected columns"):
        core.import_data(file_path) #Test the exception in case of wrong col names or order

def test_invalid_timestamps(tmp_path):
    file_path = tmp_path / "bad_time.csv"
    file_content = (
        "Time,u,v,w,T_s\n"
        "2012-09:28_00:00:00,1.0,2.0,3.0,25.0\n"
    )
    file_path.write_text(file_content)

    with pytest.raises(ValueError, match="non-valid"):
        core.import_data(file_path) #Test the exception in case of non-valid entries in the "Time" column

##### testing core.min_to_points() #####

def test_min_to_points():
    assert core.min_to_points(1, 1) == 60 # 1 min, 1 Hz
    assert core.min_to_points(10, 1) == 600 # 10 min, 1 Hz
    assert core.min_to_points(5, 2) == 600 # 5 min, 2 Hz
    assert core.min_to_points(10, 0) == 0 #  10 min, 0 Hz
    assert core.min_to_points(0, 1) == 0 # 0 min, 1 Hz


##### testing core.running_stats() #####

def test_running_stats_odd_window():
    array = np.array([1, 2, 3, 4, 5]) # test with odd window
    window_length = 3
    mean, std = core.running_stats(array, window_length)

    expected_mean = np.array([1.333, 2.0, 3.0, 4.0, 4.667])
    expected_std = np.array([0.471, 0.816, 0.816, 0.816, 0.471])

    assert_almost_equal(mean, expected_mean, decimal=3)
    assert_almost_equal(std, expected_std, decimal=3)

def test_running_stats_even_window(): #test the raising of the warning for an even window
    array = np.array([1, 2, 3, 4, 5])
    window_length = 4

    with pytest.warns(UserWarning, match="window_length is even"):
        core.running_stats(array, window_length)

def test_running_stats_with_nans():
    # Test con NaN nell'array
    array = np.array([1, 2, np.nan, 4, 5])
    window_length = 3
    mean, std = core.running_stats(array, window_length)

    expected_mean = np.array([1.333, 1.5, 3.0, 4.500, 4.667]) #[0]: 1+1+2=4, 4/3=1.33; [1]: 1+2+NaN=3, 3/2=1.5; [2]: 2+NaN+4=6, 6/2=3; [3]: NaN+4+5=9, 9/2=4.5 ; [4]: 4+5+5=14, 14/3=4.66
    expected_std = np.array([0.471, 0.5, 1, 0.5, 0.471]) 
    # [0]: (1,1,2), mean = 1.333, deviations = (-0.333, -0.333, 0.667),
    #       squared = (0.111, 0.111, 0.445), mean squared = 0.222, sqrt = 0.471

    # [1]: (1,2,nan), mean = 1.5, deviations = (-0.5, 0.5),
    #       squared = (0.25, 0.25), mean squared = 0.25, sqrt = 0.5

    # [2]: (2,nan,4), mean = 3, deviations = (-1, 1),
    #       squared = (1, 1), mean squared = 1, sqrt = 1

    # [3]: (nan,4,5), mean = 4.5, deviations = (-0.5, 0.5),
    #       squared = (0.25, 0.25), mean squared = 0.25, sqrt = 0.5

    # [4]: (4,5,5), mean = 4.667, deviations = (-0.667, 0.333, 0.333),
    #       squared = (0.445, 0.111, 0.111), mean squared = 0.222, sqrt = 0.471

    assert_almost_equal(mean, expected_mean, decimal=3)
    assert_almost_equal(std, expected_std, decimal=3)


def test_running_stats_invalid_window():
    array = np.array([1, 2, 3, 4, 5])
    
    with pytest.raises(ValueError): # zero length window
        core.running_stats(array, window_length=0)

    with pytest.raises(ValueError): # negative window length
        core.running_stats(array, window_length=-1)

    with pytest.raises(ValueError): # window is longer than the array
        core.running_stats(array, window_length=6)