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

def test_remove_beyond_threshold():
    # DataFrame fittizio pensato per coprire tutti i casi
    data = pd.DataFrame({
        'u': [5.0, 10.1, -9.9],      # solo il secondo supera la soglia
        'v': [0.0, -15.0, 10.0],     # solo il secondo supera (assoluto > 10)
        'w': [0.05, -0.1, 0.2],      # solo il terzo supera
        'T_s': [19.9, -20.0, 25.0]   # solo il terzo supera (assoluto > 20)
    }, index=pd.date_range("2022-01-01", periods=3, freq="s"))

    horizontal_threshold = 10.0
    vertical_threshold = 0.1
    temperature_threshold = 20.0

    cleaned = pre_processing.remove_beyond_threshold(
        data,
        horizontal_threshold,
        vertical_threshold,
        temperature_threshold
    )

    # Colonna "u"
    assert not np.isnan(cleaned.loc[data.index[0], 'u'])  # 5.0 < 10
    assert np.isnan(cleaned.loc[data.index[1], 'u'])      # 10.1 > 10
    assert not np.isnan(cleaned.loc[data.index[2], 'u'])  # 9.9 < 10

    # Colonna "v"
    assert not np.isnan(cleaned.loc[data.index[0], 'v'])  # 0.0 < 10
    assert np.isnan(cleaned.loc[data.index[1], 'v'])      # 15.0 > 10
    assert not np.isnan(cleaned.loc[data.index[2], 'v'])  # 10.0 == 10 → OK

    # Colonna "w"
    assert not np.isnan(cleaned.loc[data.index[0], 'w'])  # 0.05 < 0.1
    assert not np.isnan(cleaned.loc[data.index[1], 'w'])  # 0.1 == 0.1 → OK
    assert np.isnan(cleaned.loc[data.index[2], 'w'])      # 0.2 > 0.1

    # Colonna "T_s"
    assert not np.isnan(cleaned.loc[data.index[0], 'T_s'])  # 19.9 < 20
    assert not np.isnan(cleaned.loc[data.index[1], 'T_s'])  # 20.0 == 20 → OK
    assert np.isnan(cleaned.loc[data.index[2], 'T_s'])      # 25.0 > 20