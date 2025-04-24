import sys
import os
import pandas as pd
import numpy as np
import logging
from unittest.mock import MagicMock
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import src.core as core
# import src.pre_processing as pre_processing
import pre_processing
import core


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

def test_remove_beyond_threshold_basic():
    array = np.array([1.0, 5.0, -7.0, 3.0, -10.0])
    threshold = 6.0
    expected_clean = np.array([1.0, 5.0, np.nan, 3.0, np.nan])
    expected_count = 2

    result_array, count = pre_processing.remove_beyond_threshold(array, threshold)

    assert count == expected_count, "Incorrect number of values replaced"
    assert np.allclose(result_array[:2], expected_clean[:2], equal_nan=True)
    assert np.allclose(result_array[3], expected_clean[3], equal_nan=True)
    assert np.isnan(result_array[2]) and np.isnan(result_array[4]), "Values beyond threshold not replaced with NaN"

def test_remove_beyond_threshold_no_replacement():
    array = np.array([1.0, 2.0, -2.5])
    threshold = 5.0
    result_array, count = pre_processing.remove_beyond_threshold(array, threshold)

    assert count == 0, "No values should be replaced"
    assert np.array_equal(result_array, array), "Array should remain unchanged"

def test_remove_beyond_threshold_all_replaced():
    array = np.array([100.0, -200.0, 300.0])
    threshold = 50.0
    result_array, count = pre_processing.remove_beyond_threshold(array, threshold)

    assert count == 3, "All values should be replaced"
    assert np.all(np.isnan(result_array)), "All values should be NaN"

def test_remove_beyond_threshold_empty_array():
    array = np.array([])
    threshold = 10.0
    result_array, count = pre_processing.remove_beyond_threshold(array, threshold)

    assert count == 0, "Empty array should result in zero replacements"
    assert result_array.size == 0, "Result should be an empty array"

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

##### testing pre_processing.despiking_VM97() #####

def test_no_spikes():
    signal = np.ones(100)
    result = pre_processing.despiking_VM97(signal,
                            c=2.0, 
                            window_length=5, 
                            max_consecutive_spikes=3, 
                            max_iterations=5, 
                            logger=None)
    np.testing.assert_array_equal(result, signal)

def generate_data_with_spikes(size: int,
                              spike_indices: list,
                              spike_value: float) -> np.ndarray:
    data = np.random.normal(loc=1, scale=1, size=size) # values normal distributed around mean=1, with std_dev=1
    for idx in spike_indices:
        data[idx] = spike_value
    return data


def test_despiking_removes_spikes():
    # tests that the function removes peaks from a sequence with well-spaced peaks
    size =1*60*60*1 # 1 Hz signal, duration: 1h => 3600 points
    # spikes at 15 min, 30 min, 45 min after the start
    spike_indices = [900, 1800, 2700]  # idices of the spikes
    spike_value = 20  # Value possibly greater than the bounds: 1*3+1=5 (see after)
    array = generate_data_with_spikes(size,
                                      spike_indices,
                                      spike_value)
    
    # input parameters for the data creation function
    c = 3.0
    window_length = 5*60+1 # (5min window length + 1 to obtain odd number of points)
    max_consecutive_spikes = 3
    max_iterations = 10

    despiked_array = pre_processing.despiking_VM97(array,
                                    c=c,
                                    window_length=window_length,
                                    max_consecutive_spikes=max_consecutive_spikes,
                                    max_iterations=max_iterations,
                                    logger=None)
    
    for idx in spike_indices:
        assert abs(despiked_array[idx]) < 10, f"Spike at index {idx} was not removed: value {despiked_array[idx]}"


def test_despiking_preserves_normal_values():
    # tests that the function doesn't remove non-spike values, 
    size =1*60*60*1 # 1 Hz signal, duration: 1h => 3600 points
    array = generate_data_with_spikes(size, [], 0) # no spikes
    
    # Parametri di input per la funzione
    c = 5.0
    window_length = 5
    max_consecutive_spikes = 2
    max_iterations = 10
    
    # Esegui la funzione
    despiked_array = pre_processing.despiking_VM97(array,
                                    c=c,
                                    window_length=window_length,
                                    max_consecutive_spikes=max_consecutive_spikes,
                                    max_iterations=max_iterations,
                                    logger=None)
    
    assert np.allclose(array, despiked_array, atol=1e-2), "Non-spike values incorrectly modified."

def test_despiking_stops_on_max_iterations(monkeypatch):
    # Falso array con spike che non scompariranno mai
    input_array = np.array([1.0]*10 + [100.0] + [1.0]*10)

    # Mock delle funzioni esterne
    def mock_running_stats(arr, window_length):
        mean = np.ones_like(arr)  # media costante
        std = np.ones_like(arr)   # deviazione standard costante
        return mean, std

    def mock_identify_interp_spikes(arr, mask, max_consecutive_spikes):
        return arr, 1  # simula spike costanti, mai 0

    monkeypatch.setattr(core, "running_stats", mock_running_stats)
    monkeypatch.setattr("pre_processing.identify_interp_spikes", mock_identify_interp_spikes)

    mock_logger = MagicMock()
    max_iter = 3

    temp_array = pre_processing.despiking_VM97(
        array_to_despike=input_array,
        c=2.0,
        window_length=5,
        max_consecutive_spikes=1,
        max_iterations=max_iter,
        logger=mock_logger
    )

    # Verifica che il numero di iterazioni loggate corrisponda a max_iterations + 1
    logged_iterations = [call for call in mock_logger.info.call_args_list if "Iteration" in str(call)]
    assert len(logged_iterations) == max_iter + 1  # Iterazioni: 0, 1, ..., max_iter


def test_logger_usage(caplog):
    # tests that the function removes peaks from a sequence with well-spaced peaks
    size =1*60*60*1 # 1 Hz signal, duration: 1h => 3600 points
    # spikes at 15 min, 30 min, 45 min after the start
    spike_indices = [900, 1800, 2700]  # idices of the spikes
    spike_value = 20  # Value possibly greater than the bounds: 1*3+1=5 (see after)
    array = generate_data_with_spikes(size,
                                      spike_indices,
                                      spike_value)
    
    # input parameters for the data creation function
    c = 3.0
    window_length = 5*60+1 # (5min window length + 1 to obtain odd number of points)
    max_consecutive_spikes = 3
    max_iterations = 10
    
    # Esegui la funzione con il logger
    with caplog.at_level(logging.INFO):
        pre_processing.despiking_VM97(array,
                       c=c,
                       window_length=window_length,
                       max_consecutive_spikes=max_consecutive_spikes,
                       max_iterations=max_iterations,
                       logger=logging.getLogger())
    
    # Verifica che il logger abbia registrato informazioni sull'iterazione
    assert "Iteration:" in caplog.text, "Logger did not record any string 'iteration'"
