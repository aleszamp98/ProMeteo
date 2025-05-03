import pytest
from numpy.testing import assert_almost_equal
import tempfile
import textwrap
import configparser
import pandas as pd
import numpy as np
import core
import configparser

#######################################################################
#################### testing core.load_config() #######################
#######################################################################


def write_config_file(content: str) -> str:
    """
    Writes a given content to a temporary configuration file and returns its file path.

    This helper function creates a temporary file, writes the provided content into it,
    and ensures the file is flushed to disk. The file is not deleted after use, and its
    file path is returned for further processing.

    Parameters
    ----------
    content : str
        The content to be written to the temporary config file. It is expected
        to be a string, typically in INI or config format.

    Returns
    -------
    str
        The file path of the temporary configuration file that has been created.
    """
    tmp = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    tmp.write(textwrap.dedent(content))
    tmp.flush()
    return tmp.name

# --- Valid config ---
valid_config = """
[general]
rawdata_path = ./data
dir_out = ./output
sampling_freq = 20.0
model = RM_YOUNG_81000

[remove_beyond_threshold]
horizontal_threshold = 2.0
vertical_threshold = 1.5
temperature_threshold = 0.5

[despiking]
despiking_method = robust
window_length_despiking = 3.0
max_length_spike = 10
max_iterations = 5
c_H = 1.0
c_V = 1.0
c_T = 1.0
c_robust = 1.5

[averaging]
window_length_averaging = 10.0

[rotation]
reference_frame = LEC
azimuth = 90.0
"""
# --- Valid config test ---
def test_valid_config():
    """Test for a valid configuration file."""
    # Arrange: Write a valid config file
    path = write_config_file(valid_config)
    
    # Act: Load the config file using the core function
    result = core.load_config(path)
    
    # Assert: Check if the result is a dictionary
    assert isinstance(result, dict)
    
    # Assert: Verify specific config values
    assert result['model'] == 'RM_YOUNG_81000'
    assert result['sampling_freq'] == 20.0
    assert result['azimuth'] == 90.0


# --- Error handling tests ---

def test_file_not_found():
    """Test if FileNotFoundError is raised for a non-existent file."""
    # Act & Assert: Check if importing a non-existent file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        core.load_config("non_existent.ini")


def test_missing_section():
    """Test if a missing section in the config raises a NoSectionError."""
    # Arrange: Create a config with a missing section
    config = """
    [general]
    rawdata_path = ./data
    dir_out = ./output
    sampling_freq = 20.0
    model = RM_YOUNG_81000
    """
    path = write_config_file(config)
    
    # Act & Assert: Check if NoSectionError is raised for missing section
    with pytest.raises(configparser.NoSectionError):
        core.load_config(path)


def test_missing_option():
    """Test if a missing option in a section raises a NoOptionError."""
    # Arrange: Create a config with a missing option
    config = """
    [general]
    dir_out = ./output
    sampling_freq = 20.0
    model = RM_YOUNG_81000

    [remove_beyond_threshold]
    horizontal_threshold = 2.0
    vertical_threshold = 1.5
    temperature_threshold = 0.5

    [despiking]
    despiking_method = robust
    window_length_despiking = 3.0
    max_length_spike = 10
    max_iterations = 5
    c_H = 1.0
    c_V = 1.0
    c_T = 1.0
    c_robust = 1.5

    [averaging]
    window_length_averaging = 10.0

    [rotation]
    reference_frame = LEC
    azimuth = 90.0
    """
    path = write_config_file(config)
    
    # Act & Assert: Check if NoOptionError is raised for missing option
    with pytest.raises(configparser.NoOptionError):
        core.load_config(path)


def test_invalid_model():
    """Test if an invalid model raises a ValueError."""
    # Arrange: Create a config with an invalid model
    config = valid_config.replace("RM_YOUNG_81000", "INVALID_MODEL")
    path = write_config_file(config)
    
    # Act & Assert: Check if ValueError is raised for invalid model
    with pytest.raises(ValueError, match="Invalid input for 'model'"):
        core.load_config(path)


def test_invalid_despiking_method():
    """Test if an invalid despiking method raises a ValueError."""
    # Arrange: Create a config with an invalid despiking method
    config = valid_config.replace("robust", "not_valid_method", 1)
    path = write_config_file(config)
    
    # Act & Assert: Check if ValueError is raised for invalid despiking method
    with pytest.raises(ValueError, match="Invalid 'despiking_method'"):
        core.load_config(path)


def test_invalid_reference_frame():
    """Test if an invalid reference frame raises a ValueError."""
    # Arrange: Create a config with an invalid reference frame
    config = valid_config.replace("LEC", "BAD_FRAME")
    path = write_config_file(config)
    
    # Act & Assert: Check if ValueError is raised for invalid reference frame
    with pytest.raises(ValueError, match="Invalid 'reference_frame'"):
        core.load_config(path)


def test_azimuth_out_of_bounds():
    """Test if an azimuth out of bounds raises a ValueError."""
    # Arrange: Create a config with an invalid azimuth value
    config = valid_config.replace("90.0", "361.0")
    path = write_config_file(config)
    
    # Act & Assert: Check if ValueError is raised for azimuth out of bounds
    with pytest.raises(ValueError, match="azimuth must be between 0 and 360"):
        core.load_config(path)


def test_invalid_type_in_config():
    """Test if an invalid type in the config raises a ValueError."""
    # Arrange: Create a config with an invalid type for sampling_freq
    config = valid_config.replace("20.0", "not_a_float", 1)  # invalid float for sampling_freq
    path = write_config_file(config)
    
    # Act & Assert: Check if ValueError is raised for invalid type
    with pytest.raises(ValueError, match="Invalid type in config file"):
        core.load_config(path)

#######################################################################
################## testing core.import_data() #########################
#######################################################################

def test_import_data_valid_file(tmp_path):
    # Arrange: create a valid CSV file with expected columns and content
    file_path = tmp_path / "try.csv"
    file_content = (
        "Time,u,v,w,T_s\n"
        "2024-01-01 00:00:00,1.0,2.0,3.0,25.0\n"
        "2024-01-01 00:01:00,1.1,2.1,3.1,25.1\n"
    )
    file_path.write_text(file_content)
    
    # Act: import the CSV file
    df = core.import_data(file_path)
    
    # Assert: check if the output is a DataFrame
    assert isinstance(df, pd.DataFrame)
    # Assert: check if the columns match the expected ones
    assert list(df.columns) == ["u", "v", "w", "T_s"]
    # Assert: check if the index is in datetime format
    assert isinstance(df.index[0], pd.Timestamp)

def test_import_data_file_not_found():
    # Act & Assert: check if importing a non-existent file raises a FileNotFoundError
    with pytest.raises(FileNotFoundError):
        core.import_data("fail.csv")

def test_import_data_invalid_columns(tmp_path):
    # Arrange: create a CSV file with incorrect column names
    file_path = tmp_path / "bad_columns.csv"
    file_content = (
        "Timestamp,u,v,w,T_s\n"  # "Time" -> "Timestamp"
        "2024-01-01 00:00:00,1.0,2.0,3.0,25.0\n"
    )
    file_path.write_text(file_content)

    # Act & Assert: check if importing the file raises a ValueError with the expected message
    with pytest.raises(ValueError, match="Expected columns"):
        core.import_data(file_path)

def test_import_data_invalid_timestamps(tmp_path):
    # Arrange: create a CSV file with invalid timestamp format
    file_path = tmp_path / "bad_time.csv"
    file_content = (
        "Time,u,v,w,T_s\n"
        "2012-09:28_00:00:00,1.0,2.0,3.0,25.0\n"
    )
    file_path.write_text(file_content)

    # Act & Assert: check if importing the file raises a ValueError with the expected message
    with pytest.raises(ValueError, match="non-valid"):
        core.import_data(file_path)

#######################################################################
################## testing core.min_to_points() #######################
#######################################################################

def test_min_to_points():
    # Act & Assert: check the conversion of minutes to points for different cases
    
    # 1 min, 1 Hz -> 60 points (even), so should return 61
    assert core.min_to_points(1, 1) == 61  
    # 10 min, 1 Hz -> 600 points (even), so should return 601
    assert core.min_to_points(10, 1) == 601  
    # 5 min, 2 Hz -> 600 points (even), so should return 601
    assert core.min_to_points(5, 2) == 601  
    # 10 min, 0 Hz -> 0 points (even), so should return 0
    assert core.min_to_points(10, 0) == 0  
    # 0 min, 1 Hz -> 0 points (even), so should return 0
    assert core.min_to_points(0, 1) == 0  
    # 1 min, 2 Hz -> 120 points (even), so should return 121
    assert core.min_to_points(1, 2) == 121  
    # 0 min, 0 Hz -> 0 points (even), so should return 0
    assert core.min_to_points(0, 0) == 0  
    # 3 min, 5 Hz -> 900 points (odd), so should return 901
    assert core.min_to_points(3, 5) == 901  
    # 2 min, 3 Hz -> 360 points (even), so should return 361
    assert core.min_to_points(2, 3) == 361  
    # 1 min, 1 Hz -> 60 points (even), so should return 61
    assert core.min_to_points(1, 1) == 61  # Ensuring repeated case works
    # 0 min, 0 Hz -> 0 points, edge case
    assert core.min_to_points(0, 0) == 0  # Should return 0
    # 0 min, 0 Hz -> 0 points, but still ensuring it works
    assert core.min_to_points(0, 0) == 0


######################################################################
################## testing core.running_stats() ######################
######################################################################

def test_running_stats_odd_window():
    # Arrange: create an array with odd window length
    array = np.array([1, 2, 3, 4, 5])  # test with odd window
    window_length = 3
    
    # Act: calculate running mean and std
    mean, std = core.running_stats(array, window_length)

    # Expected results
    expected_mean = np.array([1.333, 2.0, 3.0, 4.0, 4.667])
    expected_std = np.array([0.471, 0.816, 0.816, 0.816, 0.471])

    # Assert: check if the mean and std are correct
    assert_almost_equal(mean, expected_mean, decimal=3)
    assert_almost_equal(std, expected_std, decimal=3)

def test_running_stats_even_window():
    # Arrange: create an array with even window length
    array = np.array([1, 2, 3, 4, 5])
    window_length = 4

    # Act & Assert: check if a warning is raised for an even window length
    with pytest.warns(UserWarning, match="window_length is even"):
        core.running_stats(array, window_length)

def test_running_stats_with_nans():
    # Arrange: create an array with NaN values
    array = np.array([1, 2, np.nan, 4, 5])
    window_length = 3
    
    # Act: calculate running mean and std
    mean, std = core.running_stats(array, window_length)

    # Expected results
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
    expected_mean = np.array([1.333, 1.5, 3.0, 4.500, 4.667])
    expected_std = np.array([0.471, 0.5, 1, 0.5, 0.471])

    # Assert: check if the mean and std are correct
    assert_almost_equal(mean, expected_mean, decimal=3)
    assert_almost_equal(std, expected_std, decimal=3)


def test_running_stats_invalid_window():
    # Arrange: create an array for invalid window length tests
    array = np.array([1, 2, 3, 4, 5])
    
    # Act & Assert: check if ValueError is raised for invalid window lengths
    with pytest.raises(ValueError):  # zero length window
        core.running_stats(array, window_length=0)

    with pytest.raises(ValueError):  # negative window length
        core.running_stats(array, window_length=-1)

    with pytest.raises(ValueError):  # window is longer than the array
        core.running_stats(array, window_length=6)

#######################################################################
############# testing core.running_stats_robust() #####################
#######################################################################

def test_running_stats_robust_odd_window():
    # Arrange: create an array with odd window length
    array = np.array([1, 2, 3, 4, 5])
    window_length = 3
    median, std_robust = core.running_stats_robust(array, window_length)

    # Arrange: create arrays for calculating median and percentiles
    arrays = [
        np.array([1, 1, 2]),
        np.array([1, 2, 3]),
        np.array([2, 3, 4]),
        np.array([3, 4, 5]),
        np.array([4, 5, 5])
    ]

    percentile_84 = np.full(5, np.nan)
    percentile_16 = np.full(5, np.nan)
    median = np.full(5, np.nan)
    for i, array in enumerate(arrays):
        median[i] = np.median(array)
        percentile_84[i] = np.percentile(array, 84)
        percentile_16[i] = np.percentile(array, 16)

    # Expected results
    expected_median = median
    expected_std_robust = 0.5 * (percentile_84 - percentile_16)

    # Assert: check if the median and robust std are correct
    assert_almost_equal(median, expected_median, decimal=3)
    assert_almost_equal(std_robust, expected_std_robust, decimal=3)

def test_running_stats_robust_even_window():
    # Arrange: create an array with even window length
    array = np.array([1, 2, 3, 4, 5])
    window_length = 4

    # Act & Assert: check if a warning is raised for an even window length
    with pytest.warns(UserWarning, match="window_length is even"):
        core.running_stats_robust(array, window_length)

def test_running_stats_robust_with_nans():
    # Arrange: create an array with NaN values
    array = np.array([1, 2, np.nan, 4, 5])
    window_length = 3
    median, std_robust = core.running_stats_robust(array, window_length)

    # Arrange: create arrays for calculating median and percentiles with NaN handling
    arrays = [
        np.array([1, 1, 2]),
        np.array([1, 2, np.nan]),
        np.array([2, np.nan, 4]),
        np.array([np.nan, 4, 5]),
        np.array([4, 5, 5])
    ]

    percentile_84 = np.full(5, np.nan)
    percentile_16 = np.full(5, np.nan)
    median = np.full(5, np.nan)
    for i, array in enumerate(arrays):
        median[i] = np.nanmedian(array)
        percentile_84[i] = np.nanpercentile(array, 84)
        percentile_16[i] = np.nanpercentile(array, 16)

    # Expected results
    expected_median = median
    expected_std_robust = 0.5 * (percentile_84 - percentile_16)

    # Assert: check if the median and robust std are correct
    assert_almost_equal(median, expected_median, decimal=3)
    assert_almost_equal(std_robust, expected_std_robust, decimal=3)

def test_running_stats_robust_invalid_window():
    # Arrange: create an array for invalid window length tests
    array = np.array([1, 2, 3, 4, 5])

    # Act & Assert: check if ValueError is raised for invalid window lengths
    with pytest.raises(ValueError):
        core.running_stats_robust(array, window_length=0)

    with pytest.raises(ValueError):
        core.running_stats_robust(array, window_length=-1)

    with pytest.raises(ValueError):
        core.running_stats_robust(array, window_length=6)


#######################################################################
#######################################################################
#######################################################################