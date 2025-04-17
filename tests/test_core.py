import pytest
import pandas as pd
# import os 
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
import core

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
