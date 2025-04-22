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
    # fictious DataFrame
    data = pd.DataFrame({
        'u': [5.0, 10.1, -9.9],      # [1] beyond threshold
        'v': [0.0, -15.0, 10.0],     # [1] beyond threshold
        'w': [0.05, -0.1, 0.2],      # [2] beyond threshold
        'T_s': [19.9, -20.0, 25.0]   # [2] beyond threshold
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

    # "u"
    assert not np.isnan(cleaned.loc[data.index[0], 'u'])  # 5.0 < 10
    assert np.isnan(cleaned.loc[data.index[1], 'u'])      # 10.1 > 10
    assert not np.isnan(cleaned.loc[data.index[2], 'u'])  # 9.9 < 10
    # "v"
    assert not np.isnan(cleaned.loc[data.index[0], 'v'])  # 0.0 < 10
    assert np.isnan(cleaned.loc[data.index[1], 'v'])      # 15.0 > 10
    assert not np.isnan(cleaned.loc[data.index[2], 'v'])  # 10.0 == 10 => OK
    # "w"
    assert not np.isnan(cleaned.loc[data.index[0], 'w'])  # 0.05 < 0.1
    assert not np.isnan(cleaned.loc[data.index[1], 'w'])  # 0.1 == 0.1 => OK
    assert np.isnan(cleaned.loc[data.index[2], 'w'])      # 0.2 > 0.1
    # "T_s"
    assert not np.isnan(cleaned.loc[data.index[0], 'T_s'])  # 19.9 < 20
    assert not np.isnan(cleaned.loc[data.index[1], 'T_s'])  # 20.0 == 20 => OK
    assert np.isnan(cleaned.loc[data.index[2], 'T_s'])      # 25.0 > 20

##### testing pre_precessing.linear_interp() #####

def test_linear_interp():
    result = pre_processing.linear_interp(0.0, 10.0, 5) # testing with valid values
    expected = np.array([1.667, 3.333, 5.0, 6.667, 8.333])
    np.testing.assert_almost_equal(result, expected, decimal=3)

    result = pre_processing.linear_interp(5.0, 10.0, 1) # testing with length = 1
    expected = np.array([7.5])
    np.testing.assert_array_equal(result, expected)

    result = pre_processing.linear_interp(5.0, 5.0, 5) # test with left_value == right_value
    expected = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    np.testing.assert_array_equal(result, expected)


##### testing pre_processing.identify_interp_spikes() #####

def test_basic_interpolation():
    array = np.array([1.0, 2.0, 100.0, 200.0, 5.0])
    mask = np.array([False, False, True, True, False])
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # interp tra 2.0 e 5.0
    max_length = 2
    result, count = pre_processing.identify_interp_spikes(array.copy(), mask, max_length)
    np.testing.assert_almost_equal(result, expected, decimal=3)
    assert count == 1

def test_no_interpolation_if_too_long():
    array = np.array([1.0, 2.0, 100.0, 200.0, 300.0, 5.0])
    mask = np.array([False, False, True, True, True, False])
    max_length = 2
    expected = array.copy()
    result, count = pre_processing.identify_interp_spikes(array.copy(), mask, max_length)
    np.testing.assert_array_equal(result, expected)
    assert count == 0

def test_no_interpolation_on_boundary():
    array = np.array([100.0, 200.0, 3.0, 4.0, 5.0])
    mask = np.array([True, True, False, False, False])
    max_length = 2
    expected = array.copy()
    result, count = pre_processing.identify_interp_spikes(array.copy(), mask, max_length)
    np.testing.assert_array_equal(result, expected)
    assert count == 0

def test_multiple_spikes():
    array = np.array([1.0, 2.0, 100.0, 200.0, 5.0, 6.0, 300.0, 7.0])
    mask = np.array([False, False, True, True, False, False, True, False])
    max_length = 2
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.5, 7.0])
    result, count = pre_processing.identify_interp_spikes(array.copy(), mask, max_length)
    np.testing.assert_almost_equal(result, expected, decimal=3)
    assert count == 2