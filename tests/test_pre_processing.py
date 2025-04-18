import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
import pre_processing

##### testing pre_processing.test_fill_missing_regular_case() #####

def test_fill_missing_regular_case():
    # Original data: samples at 1 Hz, missing one sample
    times = pd.to_datetime([
        '2023-01-01 00:00:00',
        '2023-01-01 00:00:01',
        '2023-01-01 00:00:03'  # missing second 2
    ])
    df = pd.DataFrame({'value': [1, 2, 4]}, index=times)

    result = pre_processing.fill_missing_timestamps(df, freq=1)
    expected_index = pd.date_range('2023-01-01 00:00:00', '2023-01-01 00:00:03', freq='1s')

    # Check that the index is complete
    assert all(result.index == expected_index)

    # Check that original values are preserved
    assert result.loc['2023-01-01 00:00:00', 'value'] == 1
    assert result.loc['2023-01-01 00:00:01', 'value'] == 2
    assert np.isnan(result.loc['2023-01-01 00:00:02', 'value'])
    assert result.loc['2023-01-01 00:00:03', 'value'] == 4

def test_no_missing_timestamps():
    # Data with no missing timestamps
    times = pd.date_range('2023-01-01 00:00:00', periods=4, freq='1s')
    df = pd.DataFrame({'value': [1, 2, 3, 4]}, index=times)

    result = pre_processing.fill_missing_timestamps(df, freq=1)

    # Should be unchanged
    pd.testing.assert_frame_equal(df, result)

def test_negative_frequency():
    # Test that negative frequency raises ValueError
    df = pd.DataFrame({'value': [1]}, index=[pd.to_datetime('2023-01-01')])
    try:
        pre_processing.fill_missing_timestamps(df, freq=-10)
        assert False, "Expected ValueError"
    except ValueError:
        pass

##### testing remove_beyond_threshold() #####
