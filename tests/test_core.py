import pytest
import pandas as pd
# import os 
import sys
import configparser
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
import core


##### testing core.load_config() #####

def write_config_file(tmp_path, content: str):
    config_file = tmp_path / "config.txt"
    config_file.write_text(content)
    return config_file

def test_valid_config(tmp_path):
    content = """
    [general]
    rawdata_path = ./data.csv
    dir_out = ./output/
    sampling_freq = 20
    """
    config_path = write_config_file(tmp_path, content)
    params = core.load_config(config_path)
    assert params['sampling_freq'] == 20
    assert isinstance(params['rawdata_path'], str)
    assert isinstance(params['dir_out'], str)

def test_missing_section(tmp_path):
    content = """
    [wrong_section]
    rawdata_path = ./data.csv
    dir_out = ./output/
    sampling_freq = 10
    """
    config_path = write_config_file(tmp_path, content)
    with pytest.raises(configparser.NoSectionError):
        core.load_config(config_path)

def test_missing_option(tmp_path):
    content = """
    [general]
    rawdata_path = ./data.csv
    # dir_out missing
    sampling_freq = 10
    """
    config_path = write_config_file(tmp_path, content)
    with pytest.raises(configparser.NoOptionError):
        core.load_config(config_path)

def test_invalid_type(tmp_path):
    content = """
    [general]
    rawdata_path = ./data.csv
    dir_out = ./output/
    sampling_freq = ten
    """
    config_path = write_config_file(tmp_path, content)
    with pytest.raises(ValueError, match="sampling_freq.*integer"):
        core.load_config(config_path)

def test_config_read_empty(tmp_path):
    config_path = tmp_path / "config_missing.txt"
    config = configparser.ConfigParser()
    read_files = config.read(config_path)
    assert read_files == []  # Deve restituire una lista vuota

# def test_simple_print():
#     print("Test semplice in esecuzione!")
#     assert True


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
