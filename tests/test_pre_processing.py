import sys
import os
import pytest
import pandas as pd
import numpy as np
import logging
from unittest.mock import MagicMock
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import src.core as core
# import src.pre_processing as pre_processing
import pre_processing
import core

# setting seed for the generation of random sequences in the script
np.random.seed(42)


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

def test_despiking_too_long_spikes():

    # tests if doesn't remove spikes that are longer than the parameter `max_consecutive_spikes`
    size =1*60*60*1 # 1 Hz signal, duration: 1h => 3600 points
    # unique long spike
    spike_indices = [900, 901, 902]  # idices of the spikes
    spike_value = 20  # Value possibly greater than the bounds: 1*3+1=5 (see after)
    array = generate_data_with_spikes(size,
                                      spike_indices,
                                      spike_value)
    c = 3.0
    window_length = 5*60+1 # (5min window length + 1 to obtain odd number of points)
    max_iterations = 10
    max_consecutive_spikes_1 = 3
    despiked_array_1 = pre_processing.despiking_VM97(array,
                                    c=c,
                                    window_length=window_length,
                                    max_consecutive_spikes=max_consecutive_spikes_1,
                                    max_iterations=max_iterations,
                                    logger=None)
    for idx in spike_indices:
        assert abs(despiked_array_1[idx]) < 10, f"Spike at index {idx} was not removed: value {despiked_array_1[idx]}"

    max_consecutive_spikes_2 = 2
    despiked_array_2 = pre_processing.despiking_VM97(array,
                                    c=c,
                                    window_length=window_length,
                                    max_consecutive_spikes=max_consecutive_spikes_2,
                                    max_iterations=max_iterations,
                                    logger=None)
    for idx in spike_indices:
        assert despiked_array_2[idx]==array[idx], f"Spike at index {idx} was removed even though it was longer than the 'max_consecutive_spikes' parameter "


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
    def mock_running_stats(array, window_length):
        mean = np.ones_like(array)  # media costante
        std = np.ones_like(array)   # deviazione standard costante
        return mean, std

    def mock_identify_interp_spikes(array, mask, max_consecutive_spikes):
        return array, 1  # simula spike costanti, mai 0

    monkeypatch.setattr(core, "running_stats", mock_running_stats)
    monkeypatch.setattr("pre_processing.identify_interp_spikes", mock_identify_interp_spikes)

    mock_logger = MagicMock()
    max_iter = 3

    _ = pre_processing.despiking_VM97(
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

##### testing pre_processing.despiking_robust() #####

def generate_data_with_spikes_despiking_robust(size: int,
                              spike_indices: list,
                              spike_value: float) -> np.ndarray:
    data = np.random.uniform(0, 100, size=size) 
    # an array for which the median and the percentiles are easy to compute:
    # median=50, p16=16, p84=84
    for idx in spike_indices:
        data[idx] = spike_value
    return data

def test_no_spikes_robust():
    signal = np.ones(100)
    result, _ = pre_processing.despiking_robust(signal,
                                             c=2.0,
                                             window_length=5)
    np.testing.assert_array_equal(result, signal)

def test_despiking_removes_spikes_robust():
    size = 1 * 60 * 60 * 1  # 1 Hz signal, duration: 1h => 3600 points
    spike_indices = [900, 1800, 2700]  # spikes indices 15 min, 30 min, 45 min)
    spike_value = 200  # has to be greater than the [mean + c*(p84-p16)/2] = 50 + 3*[(84-16)/2] = 152
    array = generate_data_with_spikes_despiking_robust(size,
                                                       spike_indices,
                                                       spike_value)
    
    c = 3  # Parametro c per la funzione
    window_length = 5 * 60 + 1
    
    despiked_array, count_spike = pre_processing.despiking_robust(array,
                                                                  c=c,
                                                                  window_length=window_length)
    
    # compute the running median
    running_median, _ = core.running_stats_robust(array,
                                                  window_length)

    for idx in spike_indices: # verifies if the spikes were removed: sobstituted with 
        assert despiked_array[idx] == running_median[idx], f"Spike at index {idx} was not removed: value {despiked_array[idx]}"
    
    # verifies if the number of values modified is correct
    assert count_spike == len(spike_indices), f"Expected {len(spike_indices)} spikes, but got {count_spike}"


def test_despiking_preserves_normal_values_robust():
    size = 1 * 60 * 60 * 1  # 1h=3600 points
    array = np.random.normal(loc=1, scale=1, size=size)  # without spikes
    
    c = 5.0 
    window_length = 5
    
    despiked_array, _ = pre_processing.despiking_robust(array,
                                                                  c=c,
                                                                  window_length=window_length)
    
    np.testing.assert_array_equal(array,
                                  despiked_array,
                                  err_msg="Non-spike values incorrectly modified.")
    
##### testing pre_processing.interp_nan() #####

def test_no_nans():
    array = np.array([1.0, 2.0, 3.0])
    result, count = pre_processing.interp_nan(array)
    np.testing.assert_array_equal(result, array)
    assert count == 0

def test_single_nan():
    array = np.array([1.0, np.nan, 3.0])
    expected = np.array([1.0, 2.0, 3.0])
    result, count = pre_processing.interp_nan(array)
    np.testing.assert_allclose(result, expected)
    assert count == 1

def test_multiple_nans():
    array = np.array([1.0, np.nan, np.nan, 4.0])
    expected = np.array([1.0, 2.0, 3.0, 4.0])
    result, count = pre_processing.interp_nan(array)
    np.testing.assert_allclose(result, expected)
    assert count == 2

def test_nan_at_edges():
    array = np.array([np.nan, 1.0, 2.0, np.nan])
    expected = np.array([np.nan, 1.0, 2.0, np.nan])
    result, count = pre_processing.interp_nan(array)
    np.testing.assert_array_equal(result, expected)
    assert count == 0

def test_all_nans():
    array = np.array([np.nan, np.nan])
    expected = np.array([np.nan, np.nan])
    result, count = pre_processing.interp_nan(array)
    np.testing.assert_array_equal(result, expected)
    assert count == 0


##### testing pre_processing.rotation_to_LEC_reference() #####

def test_invalid_azimuth_low():
    wind = np.zeros((3, 10))
    # lower than 0
    azimuth = -10
    model = "RM_YOUNG_81000"
    with pytest.raises(ValueError, match="azimuth is outside the range"):
        pre_processing.rotation_to_LEC_reference(wind, azimuth, model)
    # higher than 360
    azimuth = 400
    model = "CAMPBELL_CSAT3"
    with pytest.raises(ValueError, match="azimuth is outside the range"):
        pre_processing.rotation_to_LEC_reference(wind, azimuth, model)

def test_invalid_model():
    wind = np.zeros((3, 10))
    azimuth = 0
    model = "BEST_ANEMOMETER_EVER"
    with pytest.raises(ValueError, match="Unknown model"):
        pre_processing.rotation_to_LEC_reference(wind, azimuth, model)

def test_rotation_RM_YOUNG_81000_azimuth_zero():
    wind = np.array([
        [ 2, -2,  2,  2],
        [ 3,  3, -3,  3],
        [ 4,  4,  4, -4]
    ])
    azimuth = 0
    model = "RM_YOUNG_81000"
    
    result = pre_processing.rotation_to_LEC_reference(wind, azimuth, model)

    expected = np.array([
        -wind[0,:],  # x component inverted
        -wind[1,:],  # y component inverted
         wind[2,:]   # z component unchanged
    ])
    np.testing.assert_array_equal(result, expected, err_msg="Something wrong...")

def test_rotation_CAMPBELL_CSAT3_azimuth_zero():
    wind = np.array([
        [ 2, -2,  2,  2],
        [ 3,  3, -3,  3],
        [ 4,  4,  4, -4]
    ])
    azimuth = 0
    model = "CAMPBELL_CSAT3"
    
    result = pre_processing.rotation_to_LEC_reference(wind, azimuth, model)
    
    expected = np.array([
         wind[1,:],   # new x = old y
        -wind[0,:],   # new y = -old x
         wind[2,:]   # z component unchanged
    ])
    
    np.testing.assert_array_equal(result, expected, err_msg="Something wrong...")


def test_rotation_RM_YOUNG_81000_with_azimuth():
    wind = np.array([
        [ 2, -2,  2,  2],
        [ 3,  3, -3,  3],
        [ 4,  4,  4, -4]
    ])
    azimuth = 43
    model = "RM_YOUNG_81000"
    
    result = pre_processing.rotation_to_LEC_reference(wind, azimuth, model)

    # rotation matrix definition
    azimuth = np.deg2rad(azimuth) # degree to radians conversion
    rot_azimuth = np.zeros((3,3))
    rot_azimuth[0, 0] =  np.cos(azimuth) # input have to be angles in radians
    rot_azimuth[0, 1] =  np.sin(azimuth)
    rot_azimuth[1, 0] = -np.sin(azimuth)
    rot_azimuth[1, 1] =  np.cos(azimuth)
    rot_azimuth[2, 2] =  1

    wind_LEC = np.array([
        -wind[0,:],  # x component inverted
        -wind[1,:],  # y component inverted
         wind[2,:]  # z component unchanged
    ])
    expected = rot_azimuth @ wind_LEC
    np.testing.assert_almost_equal(result, expected, decimal=4, err_msg="Something wrong...")

def test_rotation_CAMPBELL_CSAT3_with_azimuth():
    wind = np.array([
        [ 2, -2,  2,  2],
        [ 3,  3, -3,  3],
        [ 4,  4,  4, -4]
    ])
    azimuth = 276
    model = "CAMPBELL_CSAT3"
    
    result = pre_processing.rotation_to_LEC_reference(wind, azimuth, model)
    
    azimuth = np.deg2rad(azimuth) # degree to radians conversion
    rot_azimuth = np.zeros((3,3))
    rot_azimuth[0, 0] =  np.cos(azimuth) # input have to be angles in radians
    rot_azimuth[0, 1] =  np.sin(azimuth)
    rot_azimuth[1, 0] = -np.sin(azimuth)
    rot_azimuth[1, 1] =  np.cos(azimuth)
    rot_azimuth[2, 2] =  1

    wind_LEC = np.array([
         wind[1,:],   # new x = old y
        -wind[0,:],   # new y = -old x
         wind[2,:]   # z component unchanged
    ])

    expected = rot_azimuth @ wind_LEC
    
    np.testing.assert_array_equal(result, expected, err_msg="Something wrong...")

##### testing pre_processing.rotation_to_streamline_reference() #####

def test_basic_rotation():
    # test a case where no rotation should be applied
    # Wind aligned with the x-axis
    wind = np.array([
        [1, 2, 3, 4],  # u component
        [0, 0, 0, 0],  # v component
        [0, 0, 0, 0],  # w component
    ])
    wind_averaged = np.array([
        [1, 2, 3, 4],  # mean u component
        [0, 0, 0, 0],  # mean v component
        [0, 0, 0, 0],  # mean w component
    ])

    wind_rotated = pre_processing.rotation_to_streamline_reference(wind, wind_averaged)

    # Expect no rotation: the output should match the input
    np.testing.assert_allclose(wind_rotated, wind, atol=1e-8)

def test_shape_mismatch_error():
    # Test that a ValueError is raised if the input shape is incorrect (wrong first dimension).
    # Wind has wrong first dimension (2 instead of 3)
    wind = np.array([
        [1, 2, 3, 4],
        [0, 0, 0, 0],
    ])
    wind_averaged = np.array([
        [1, 2, 3, 4],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])

    with pytest.raises(ValueError, match="must have shape"):
        pre_processing.rotation_to_streamline_reference(wind, wind_averaged)

def test_column_mismatch_error():
    # Test that a ValueError is raised if the number of columns does not match.
    # Wind has 4 columns, wind_averaged has 5 columns
    wind = np.array([
        [1, 2, 3, 4],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    wind_averaged = np.array([
        [1, 2, 3, 4, 5],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])

    with pytest.raises(ValueError, match="same number of columns"):
        pre_processing.rotation_to_streamline_reference(wind, wind_averaged)

def test_output_shape():
    # Test that the output has the correct shape (3, 4).
    wind = np.array([
        [1, 2, 3, 4],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ])
    wind_averaged = np.array([
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])

    wind_rotated = pre_processing.rotation_to_streamline_reference(wind, wind_averaged)

    assert wind_rotated.shape == (3, 4)

# def test_rotation_with_vertical_wind():
#     # Test a case where the mean wind is purely vertical. (?)
#     # Wind is purely vertical at first sample
#     wind = np.array([
#         [0, 0, 0, 0],
#         [0, 0, 0, 0],
#         [1, 0, 0, 0],
#     ])
#     wind_averaged = np.array([
#         [0, 1, 1, 1],
#         [0, 0, 0, 0],
#         [1, 0, 0, 0],
#     ])

#     wind_rotated = pre_processing.rotation_to_streamline_reference(wind, wind_averaged)

#     # The rotated x-component of the first sample should be 1
#     np.testing.assert_allclose(wind_rotated[0, 0], 1.0, atol=1e-8)

def test_rotation_with_y_vent():
    # Test a case where the wind is purely along the y-axis.

    # Wind is purely along the y-axis: (0, v, 0) for each sample
    wind = np.array([
        [0, 0, 0, 0],  # u component
        [1, 2, 3, 4],  # v component (non-zero along y-axis)
        [0, 0, 0, 0],  # w component
    ])
    
    # Averaged wind is purely horizontal along y (same for each sample)
    wind_averaged = np.array([
        [0, 0, 0, 0],  # mean u component (no horizontal wind in x)
        [1, 2, 3, 4],  # mean v component (aligned along y-axis)
        [0, 0, 0, 0],  # mean w component (no vertical wind)
    ])

    wind_rotated = pre_processing.rotation_to_streamline_reference(wind, wind_averaged)

    # After rotation, the x component should match the v component in magnitude
    np.testing.assert_allclose(wind_rotated[0, :], wind_averaged[1, :], atol=1e-8)
    
    # The y and z components should be zero
    np.testing.assert_allclose(wind_rotated[1, :], 0, atol=1e-8)
    np.testing.assert_allclose(wind_rotated[2, :], 0, atol=1e-8)

##### testing pre_processing.wind_dir_LEC_reference() #####

def test_wind_direction_scalar():
    u = 10
    v = 10
    result = pre_processing.wind_dir_LEC_reference(u, v)
    expected = 225  # 270-45 o 180+45, wind from S-W
    assert np.isclose(result, expected, rtol=1e-5)

def test_shape_mismatch():
    u = np.array([1, 2, 3])
    v = np.array([1, 2])
    with pytest.raises(ValueError, match="Shape mismatch"):
        pre_processing.wind_dir_LEC_reference(u, v)

def test_wind_directions():
    u = [ 0, -1/np.sqrt(2), -1, -1/np.sqrt(2), 0,
          1/np.sqrt(2), 1,  1/np.sqrt(2)]
    v = [-1, -1/np.sqrt(2),  0,  1/np.sqrt(2), 1, 
          1/np.sqrt(2), 0, -1/np.sqrt(2)]
    wind_dir_expected = [0, 45, 90, 135, 180, 225, 270, 315]

    result = pre_processing.wind_dir_LEC_reference(u, v)
    np.testing.assert_allclose(result, wind_dir_expected, rtol=1e-5)

##### testing pre_processing.wind_dir_modeldependent_reference() #####

def test_wind_direction_scalar_modeldependent():
    u = 10
    v = 10
    azimuth = 0.0

    # Test per modello RM_YOUNG_81000
    result_rm = pre_processing.wind_dir_modeldependent_reference(u, v, azimuth, model="RM_YOUNG_81000")
    expected_rm = 45  # inverts u and v, so from NE
    assert np.isclose(result_rm, expected_rm, rtol=1e-5)

    # Test per modello CAMPBELL_CSAT3
    result_cs = pre_processing.wind_dir_modeldependent_reference(u, v, azimuth, model="CAMPBELL_CSAT3")
    expected_cs = 315.0  # perché v_LEC = -10, u_LEC = 10 => (-10, 10) -> 315°
    assert np.isclose(result_cs, expected_cs, rtol=1e-5)

def test_shape_mismatch_modeldependent():
    u = np.array([1, 2, 3])
    v = np.array([1, 2])
    azimuth = 0.0

    with pytest.raises(ValueError, match="Shape mismatch"):
        pre_processing.wind_dir_modeldependent_reference(u, v, azimuth, model="RM_YOUNG_81000")

def test_unknown_model():
    u = [0]
    v = [1]
    azimuth = 0.0

    with pytest.raises(ValueError, match="Unknown model"):
        pre_processing.wind_dir_modeldependent_reference(u, v, azimuth, model="UNKNOWN_MODEL")

def test_wind_directions_modeldependent():
    u = [ 0, -1/np.sqrt(2), -1, -1/np.sqrt(2), 0,
          1/np.sqrt(2), 1,  1/np.sqrt(2)]
    v = [-1, -1/np.sqrt(2),  0,  1/np.sqrt(2), 1, 
          1/np.sqrt(2), 0, -1/np.sqrt(2)]
    azimuth = 0.0

    # RM_YOUNG_81000
    result_rm = pre_processing.wind_dir_modeldependent_reference(u, v, azimuth, model="RM_YOUNG_81000")
    expected_rm = [(angle + 180) % 360 for angle in [0, 45, 90, 135, 180, 225, 270, 315]]
    np.testing.assert_allclose(result_rm, expected_rm, rtol=1e-5)

    # CAMPBELL_CSAT3
    result_cs = pre_processing.wind_dir_modeldependent_reference(u, v, azimuth, model="CAMPBELL_CSAT3")
    expected_cs = [(angle + 90) % 360 for angle in [0, 45, 90, 135, 180, 225, 270, 315]]
    np.testing.assert_allclose(result_cs, expected_cs, rtol=1e-5)

def test_wind_direction_with_azimuth():
    u = [0, -1/np.sqrt(2), -1, -1/np.sqrt(2), 0,
         1/np.sqrt(2), 1, 1/np.sqrt(2)]
    v = [-1, -1/np.sqrt(2), 0, 1/np.sqrt(2), 1, 
         1/np.sqrt(2), 0, -1/np.sqrt(2)]
    azimuth = 30.0  # Rotazione di 30° (strumento montato ruotato di 30°)

    # RM_YOUNG_81000
    result_rm = pre_processing.wind_dir_modeldependent_reference(u, v, azimuth, model="RM_YOUNG_81000")
    expected_rm = [((angle + 180) - azimuth) % 360 for angle in [0, 45, 90, 135, 180, 225, 270, 315]]
    np.testing.assert_allclose(result_rm, expected_rm, rtol=1e-5)

    # CAMPBELL_CSAT3
    result_cs = pre_processing.wind_dir_modeldependent_reference(u, v, azimuth, model="CAMPBELL_CSAT3")
    expected_cs = [((angle + 90) - azimuth) % 360 for angle in [0, 45, 90, 135, 180, 225, 270, 315]]
    np.testing.assert_allclose(result_cs, expected_cs, rtol=1e-5)