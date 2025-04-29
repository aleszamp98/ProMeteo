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

#######################################################################
####### testing pre_processing.test_fill_missing_regular_case() #######
#######################################################################

def test_fill_missing_timestamps_regular_case():
    # Arrange: create a DataFrame with a missing timestamp at second 2, with sampling freq of 1 Hz
    times = pd.to_datetime([
        '2023-01-01 00:00:00',
        '2023-01-01 00:00:01',
        '2023-01-01 00:00:03'  # missing second 2
    ])
    df = pd.DataFrame({'value': [1, 2, 4]}, index=times)
    sampling_freq = 1

    # Act: fill missing timestamps
    result = pre_processing.fill_missing_timestamps(df, sampling_freq)
    expected_index = pd.date_range('2023-01-01 00:00:00', '2023-01-01 00:00:03', freq='1s')
    
    # Assert: the index is complete
    assert all(result.index == expected_index) 
    # Assert: original values are preserved
    assert result.loc['2023-01-01 00:00:00', 'value'] == 1
    assert result.loc['2023-01-01 00:00:01', 'value'] == 2
    assert np.isnan(result.loc['2023-01-01 00:00:02', 'value'])
    assert result.loc['2023-01-01 00:00:03', 'value'] == 4

def test_fill_missing_timestamps_no_missing_timestamps():
    # Arrange: create a DataFrame with with no missing timestamps
    times = pd.date_range('2023-01-01 00:00:00', periods=4, freq='1s')
    df = pd.DataFrame({'value': [1, 2, 3, 4]}, index=times)
    sampling_freq = 1

    # Act : fill missing timestamps (nothing to do in this case)
    result = pre_processing.fill_missing_timestamps(df, sampling_freq)

    # Assert: output should be equal to the input DataFrame
    pd.testing.assert_frame_equal(df, result)

def test_fill_missing_timestamps_negative_frequency():
    # Arrange: create a DataFrame with with no missing timestamps
    df = pd.DataFrame({'value': [1]}, index=[pd.to_datetime('2023-01-01')])
    sampling_freq = -1

    # Assert: 
    with pytest.raises(ValueError, match="positive"):
        pre_processing.fill_missing_timestamps(df, sampling_freq)

#######################################################################
################ testing remove_beyond_threshold() ####################
#######################################################################

def test_remove_beyond_threshold_regular_case():
    # Arrange: create an array with values smaller and bigger than the threshold
    threshold = 6.0
    array = np.array([1.0, 5.0, -7.0, 3.0, -10.0])
    # Arrange: expected outputs
    expected_clean = np.array([1.0, 5.0, np.nan, 3.0, np.nan])
    expected_count = 2

    # Act
    result_array, count = pre_processing.remove_beyond_threshold(array, threshold)

    # Assert: count of the exceeding values is as expected
    assert count == expected_count, "Incorrect number of values replaced"
    # Assert: replacement with NaNs executed in the correct places
    assert np.allclose(result_array[:2], expected_clean[:2], equal_nan=True)
    assert np.allclose(result_array[3], expected_clean[3], equal_nan=True)
    # Assert: values under threshold are preserved
    assert np.isnan(result_array[2]) and np.isnan(result_array[4]), "Values beyond threshold not replaced with NaN"

def test_remove_beyond_threshold_no_replacement():
    # Arrange: array with no values over threshold
    threshold = 5.0
    array = np.array([1.0, 2.0, -2.5])

    # Act:
    result_array, count = pre_processing.remove_beyond_threshold(array, threshold)

    # Assert: `count` should be 0 and output array should remain unchanged
    assert count == 0, "No values should be replaced"
    assert np.array_equal(result_array, array), "Array should remain unchanged"

def test_remove_beyond_threshold_all_replaced():
    # Arrange: array with all values over threshold
    threshold = 50.0
    array = np.array([100.0, -200.0, 300.0])

    # Act: all values should be replaced
    result_array, count = pre_processing.remove_beyond_threshold(array, threshold)

    # Assert: `count` has to be equal to the length of the input array
    assert count == 3, "All values should be replaced"
    # Assert: output array must contain all NaN values
    assert np.all(np.isnan(result_array)), "All values should be NaN"

def test_remove_beyond_threshold_empty_array():
    # Arrange: create empty array
    threshold = 10.0
    array = np.array([])

    # Act: no action expected
    result_array, count = pre_processing.remove_beyond_threshold(array, threshold)

    # Assert: Empty array should result in zero replacements and 0 exceeding values counted
    assert count == 0, "Empty array should result in zero replacements"
    assert result_array.size == 0, "Result should be an empty array"

def test_remove_beyond_threshold_negative_threshold():
    # Arrange: a valid array, but a negative threshold
    threshold = -6.0
    array = np.array([1.0, 5.0, -7.0, 3.0, -10.0])

    # Act & Assert: expect a ValueError due to the negative threshold
    with pytest.raises(ValueError, match="positive"):
         result_array, count = pre_processing.remove_beyond_threshold(array, threshold)

#######################################################################
############## testing pre_precessing.linear_interp() #################
#######################################################################

def test_linear_interp_regular_case():
    # Arrange: valid inputs
    left_value = 0.0
    right_value = 10.0
    length = 5
    # Arrange: expected values
    expected = np.array([1.667, 3.333, 5.0, 6.667, 8.333])
    
    # Act: linear interpolate a number of `length` values between `left_value` and `right_value`
    result = pre_processing.linear_interp(left_value, right_value, length)

    # Assert: output is as expected
    np.testing.assert_almost_equal(result, expected, decimal=3)

def test_linear_interp_length_one():
    # Arrange: valid inputs, unitary length
    left_value = 5.0
    right_value = 10.0
    length = 1
    # Arrange: expected value (the mean between left value and right value)
    expected = np.array([(left_value+right_value)/2])

    # Act:
    result = pre_processing.linear_interp(left_value, right_value, length) # testing with length = 1

    # Assert: output is as expected
    np.testing.assert_array_equal(result, expected)

def test_linear_interp_coincindent_left_right():
    # Arrange: valid inputs, right_value == left_value
    left_value = 5.0
    right_value = left_value
    length = 6
    # Arrange: expected values
    expected = np.full(length, left_value) # array with length == `length`, values all equal to left_value

    # Act:
    result = pre_processing.linear_interp(left_value, right_value, length)
    
    # Assert: output is as expected
    np.testing.assert_array_equal(result, expected)

def test_linear_interp_invalid_length():
    # Arrange: valid left and right values
    left_value = 0.0
    right_value = 1.0

    # Act & Assert: check ValueError is raised for length = 0
    length = 0
    with pytest.raises(ValueError, match="positive"):
        pre_processing.linear_interp(left_value, right_value, length)

    # Act & Assert: check ValueError is raised for negative length
    length = -1
    with pytest.raises(ValueError, match="positive"):
        pre_processing.linear_interp(left_value, right_value, length)

    # Act & Assert: check ValueError is raised for non-integer length
    length = 3.5
    with pytest.raises(ValueError, match="positive"):
        pre_processing.linear_interp(left_value, right_value, length)

#######################################################################
########### testing pre_processing.identify_interp_spikes() ###########
#######################################################################

def test_identify_interp_spikes_one_spike():
    # Arrange: spikes have to be of this length at maximum
    max_length_spike = 2
    # Arrange: array with one spike (two high values respect to the others)
    array = np.array([1.0, 2.0, 100.0, 200.0, 5.0])
    # Arrange: a mask telling where the spikes are (in the main the mask is based on running_stats)
    mask = np.array([False, False, True, True, False])
    # Arrange: expected output, [2] and [3] are interpolated values between 2.0 and 5.0
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Act:
    result, count = pre_processing.identify_interp_spikes(array.copy(), mask, max_length_spike)

    # Assert: one spike replaced with interpolated values
    np.testing.assert_almost_equal(result, expected, decimal=3)
    assert count == 1

def test_identify_interp_spikes_multiple_spikes():
    # Arrange: spikes have to be of this length at maximum
    max_length_spike = 2
    # Arrange: array containing multiple valid spikes 
    # (first spike's length: 1 < max_length, second spike of length 2 == max_length_spike)
    array = np.array([1.0, 2.0, 100.0, 200.0, 5.0, 6.0, 300.0, 7.0])
    # Arrange: a mask telling where the spikes are (in the main the mask is based on running_stats)
    mask = np.array([False, False, True, True, False, False, True, False])
    # Arrange: expected output, 
    # [2] and [3] are interpolated values between 2.0 and 5.0
    # [6] is interpolated between 6 and 7
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.5, 7.0])

    # Act:
    result, count = pre_processing.identify_interp_spikes(array.copy(), mask, max_length_spike)

    # Assert: two spikes replaced with interpolated values
    np.testing.assert_almost_equal(result, expected, decimal=3)
    assert count == 2

def test_identify_interp_spikes_no_interpolation_if_too_long():
    # Arrange: spikes have to be of this length at maximum
    max_length_spike = 2
    # Arrange: array with one spike exceeding length limit (3 > 2)
    array = np.array([1.0, 2.0, 100.0, 200.0, 300.0, 5.0])
    # Arrange: a mask telling where the spikes are (in the main the mask is based on running_stats)
    mask = np.array([False, False, True, True, True, False])
    # Arrange: expected output == input
    expected = array.copy()

    # Act:
    result, count = pre_processing.identify_interp_spikes(array.copy(), mask, max_length_spike)

    # Assert: no spike counted, input unchanged
    np.testing.assert_array_equal(result, expected)
    assert count == 0

def test_identify_interp_spikes_no_interpolation_on_boundary():
    # Arrange: spikes have to be of this length at maximum
    max_length_spike = 2
    # Arrange: array with spikes near the left edge
    array = np.array([100.0, 200.0, 3.0, 4.0, 5.0])
    # Arrange: a mask telling where the spikes are (in the main the mask is based on running_stats)
    mask = np.array([True, True, False, False, False])
    # Arrange: expected output == input
    expected = array.copy()

    # Act:
    result, count = pre_processing.identify_interp_spikes(array.copy(), mask, max_length_spike)

    # Assert: no spike counted, input unchanged
    np.testing.assert_array_equal(result, expected)
    assert count == 0

def test_identify_interp_spikes_errors():
    # Arrange: prepare valid array and mask
    array = np.array([1.0, 2.0, 3.0, 4.0])
    mask_valid = np.array([False, True, False, False])
    max_length_valid = 1

    # Act & Assert: check ValueError is raised when array and mask have different lengths
    mask_wrong_length = np.array([False, True])
    with pytest.raises(ValueError, match="same length"):
        pre_processing.identify_interp_spikes(array, mask_wrong_length, max_length_valid)

    # Act & Assert: check ValueError is raised when mask is not boolean
    mask_not_boolean = np.array([0, 1, 0, 0])
    with pytest.raises(ValueError, match="boolean array"):
        pre_processing.identify_interp_spikes(array, mask_not_boolean, max_length_valid)

    # Act & Assert: check ValueError is raised for non-integer max_length_spike
    max_length_invalid = 2.5
    with pytest.raises(ValueError, match="positive integer"):
        pre_processing.identify_interp_spikes(array, mask_valid, max_length_invalid)

    # Act & Assert: check ValueError is raised for negative max_length_spike
    max_length_invalid = -1
    with pytest.raises(ValueError, match="positive integer"):
        pre_processing.identify_interp_spikes(array, mask_valid, max_length_invalid)

    # Act & Assert: check ValueError is raised for zero max_length_spike
    max_length_invalid = 0
    with pytest.raises(ValueError, match="positive integer"):
        pre_processing.identify_interp_spikes(array, mask_valid, max_length_invalid)


#######################################################################
############## testing pre_processing.despiking_VM97() ################
#######################################################################

def generate_normal_data_with_spikes(size: int,
                                    spike_indices: list,
                                    spike_value: float) -> np.ndarray:
    """
    Generate normally distributed data with injected spikes.

    Parameters
    ----------
    size : int
        Number of data points to generate.
    spike_indices : list of int
        Indices at which to insert spike values.
    spike_value : float
        The value to assign at the specified spike indices.

    Returns
    -------
    data : np.ndarray
        1D array of normally distributed data with injected spikes.
        The normal distribution has mean=1 and standard deviation=1.
    """
    # Generate data from a normal distribution with mean=1 and std=1
    data = np.random.normal(loc=1.0, scale=1.0, size=size)
    
    # Insert spikes at the specified indices
    for idx in spike_indices:
        data[idx] = spike_value

    return data

def test_despiking_VM97_no_spikes():
    # Arrange: flat signal with no spikes
    signal = np.ones(100)
    # Arrange: expected output == input (no changes expected)
    expected = signal.copy()

    # Act: run the despiking algorithm with default parameters
    result = pre_processing.despiking_VM97(
        signal,
        c=2.0,
        window_length=5,
        max_consecutive_spikes=3,
        max_iterations=5,
        logger=None
    )

    # Assert: input unchanged, no spike removal
    np.testing.assert_array_equal(result, expected)

def test_despiking_VM97_regular_case():
    # Arrange: 1 Hz signal, duration: 1 hour => 3600 points
    size = 1 * 60 * 60 * 1
    # Arrange: spikes at 15 min, 30 min, 45 min after the start
    spike_indices = [900, 1800, 2700]
    # Arrange: spike value set intentionally above expected threshold
    spike_value = 20
    # Arrange: generate signal with spikes
    array = generate_normal_data_with_spikes(size, 
                                             spike_indices, 
                                             spike_value)
    
    # Arrange: despiking parameters
    c = 3.0
    window_length = 5 * 60 + 1  # 5-minute window, ensure odd number of points
    max_consecutive_spikes = 3
    max_iterations = 10

    # Act: apply the despiking algorithm
    despiked_array = pre_processing.despiking_VM97(
        array,
        c=c,
        window_length=window_length,
        max_consecutive_spikes=max_consecutive_spikes,
        max_iterations=max_iterations,
        logger=None
    )

    # Assert: spikes are removed (values well below threshold)
    for idx in spike_indices:
        assert abs(despiked_array[idx]) < 10, f"Spike at index {idx} was not removed: value {despiked_array[idx]}"

def test_despiking_VM97_spike_length_equal_max_spike_length():
    # Arrange: 1 Hz signal, duration: 1 hour => 3600 points
    size = 1 * 60 * 60 * 1
    # Arrange: one single long spike of 3 consecutive points
    spike_indices = [900, 901, 902]
    # Arrange: spike value intentionally above detection threshold
    spike_value = 20
    # Arrange: generate signal with long spike
    array = generate_normal_data_with_spikes(size, 
                                             spike_indices, 
                                             spike_value)

    # Arrange: despiking parameters with spike length equal to max allowed
    c = 3.0
    window_length = 5 * 60 + 1
    max_consecutive_spikes = 3
    max_iterations = 10

    # Act: apply despiking
    despiked_array = pre_processing.despiking_VM97(
        array,
        c=c,
        window_length=window_length,
        max_consecutive_spikes=max_consecutive_spikes,
        max_iterations=max_iterations,
        logger=None
    )

    # Assert: spike is removed since length == max_consecutive_spikes
    for idx in spike_indices:
        assert abs(despiked_array[idx]) < 10, f"Spike at index {idx} was not removed: value {despiked_array[idx]}"


def test_despiking_VM97_too_long_spike():
    # Arrange: 1 Hz signal, duration: 1 hour => 3600 points
    size = 1 * 60 * 60 * 1
    # Arrange: one single long spike of 3 consecutive points
    spike_indices = [900, 901, 902]
    # Arrange: spike value intentionally above detection threshold
    spike_value = 20
    # Arrange: generate signal with long spike
    array = generate_normal_data_with_spikes(size, 
                                             spike_indices, 
                                             spike_value)

    # Arrange: despiking parameters with spike length > max allowed
    c = 3.0
    window_length = 5 * 60 + 1
    max_consecutive_spikes = 2
    max_iterations = 10

    # Act: apply despiking
    despiked_array = pre_processing.despiking_VM97(
        array,
        c=c,
        window_length=window_length,
        max_consecutive_spikes=max_consecutive_spikes,
        max_iterations=max_iterations,
        logger=None
    )

    # Assert: spike remains untouched since it's too long to be removed
    for idx in spike_indices:
        assert despiked_array[idx] == array[idx], (
            f"Spike at index {idx} was removed even though it was longer than "
            f"the 'max_consecutive_spikes' parameter"
        )

def test_despiking_VM97_preserves_normal_values():
    # Arrange: 1 Hz signal, duration: 1 hour => 3600 points
    size = 1 * 60 * 60 * 1
    # Arrange: no spikes in the input array
    array = generate_normal_data_with_spikes(size, 
                                             [], 
                                             0)

    # Arrange: despiking parameters
    c = 5.0
    window_length = 5
    max_consecutive_spikes = 2
    max_iterations = 10

    # Act:
    despiked_array = pre_processing.despiking_VM97(
        array,
        c=c,
        window_length=window_length,
        max_consecutive_spikes=max_consecutive_spikes,
        max_iterations=max_iterations,
        logger=None
    )

    # Assert: input unchanged since no spikes are present
    assert np.allclose(array, despiked_array, atol=1e-2), "Non-spike values incorrectly modified."

def test_despiking_VM97_stops_on_max_iterations(monkeypatch):
    # Arrange: input array with a persistent spike
    input_array = np.array([1.0]*10 + [100.0] + [1.0]*10)

    # Arrange: mock mean and std to force detection of a spike in every iteration
    def mock_running_stats(array, window_length):
        mean = np.ones_like(array)
        std = np.ones_like(array)
        return mean, std

    # Arrange: mock spike identifier to always report a spike
    def mock_identify_interp_spikes(array, mask, max_consecutive_spikes):
        return array, 1  # one spike per iteration

    monkeypatch.setattr(core, "running_stats", mock_running_stats)
    monkeypatch.setattr("pre_processing.identify_interp_spikes", mock_identify_interp_spikes)

    # Arrange: mock logger to capture log calls
    mock_logger = MagicMock()
    max_iter = 3

    # Act:
    _ = pre_processing.despiking_VM97(
        array_to_despike=input_array,
        c=2.0,
        window_length=5,
        max_consecutive_spikes=1,
        max_iterations=max_iter,
        logger=mock_logger
    )

    # Assert: logger recorded exactly max_iterations + 1 iterations (including 0)
    logged_iterations = [call for call in mock_logger.info.call_args_list if "Iteration" in str(call)]
    assert len(logged_iterations) == max_iter + 1, "Did not stop after reaching max_iterations"


def test_despiking_VM97_logger_usage(caplog):
    # Arrange: 1 Hz signal, duration: 1 hour => 3600 points
    size = 1 * 60 * 60 * 1
    # Arrange: spikes inserted at 15, 30, and 45 minutes
    spike_indices = [900, 1800, 2700]
    spike_value = 20
    array = generate_normal_data_with_spikes(size,
                                             spike_indices,
                                             spike_value)

    # Arrange: despiking parameters
    c = 3.0
    window_length = 5 * 60 + 1
    max_consecutive_spikes = 3
    max_iterations = 10

    # Act: capture logging output
    with caplog.at_level(logging.INFO):
        pre_processing.despiking_VM97(
            array,
            c=c,
            window_length=window_length,
            max_consecutive_spikes=max_consecutive_spikes,
            max_iterations=max_iterations,
            logger=logging.getLogger()
        )

    # Assert: log includes information about iterations
    assert "Iteration:" in caplog.text, "Logger did not record any string 'Iteration:'"

def test_despiking_VM97_errors():
    # Arrange: create a dummy input array
    array = np.array([1.0, 2.0, 100.0, 2.0, 1.0])
    valid_logger = logging.getLogger("test_logger")

    # Act & Assert: check ValueError for non-positive c
    with pytest.raises(ValueError, match="positive number"):
        pre_processing.despiking_VM97(array, c=0, window_length=3, max_consecutive_spikes=1, max_iterations=5)

    # Act & Assert: check ValueError for non-positive window_length
    with pytest.raises(ValueError, match="positive integer"):
        pre_processing.despiking_VM97(array, c=1.0, window_length=0, max_consecutive_spikes=1, max_iterations=5)

    # Act & Assert: check ValueError for non-positive max_consecutive_spikes
    with pytest.raises(ValueError, match="positive integer"):
        pre_processing.despiking_VM97(array, c=1.0, window_length=3, max_consecutive_spikes=0, max_iterations=5)

    # Act & Assert: check ValueError for non-positive max_iterations
    with pytest.raises(ValueError, match="positive integer"):
        pre_processing.despiking_VM97(array, c=1.0, window_length=3, max_consecutive_spikes=1, max_iterations=0)

#######################################################################
########### testing pre_processing.despiking_robust() #################
#######################################################################

def generate_uniform_data_with_spikes(size: int,
                                      spike_indices: list,
                                      spike_value: float) -> np.ndarray:
    """
    Generate uniformly distributed data with artificial spikes, 
    designed for testing robust despiking methods.

    Parameters
    ----------
    size : int
        Number of data points to generate.
    spike_indices : list of int
        Indices at which to insert spike values.
    spike_value : float
        The value to assign at the specified spike indices.

    Returns
    -------
    data : np.ndarray
        1D array of uniformly distributed data with injected spikes.
        The uniform distribution is over the interval [0, 100], 
        so the theoretical median is 50, and the 16th and 84th 
        percentiles are approximately 16 and 84, respectively.
    """
    # Generate data from a uniform distribution in [0, 100]
    data = np.random.uniform(low=0.0, high=100.0, size=size)

    # Insert spikes at the specified indices
    for idx in spike_indices:
        data[idx] = spike_value

    return data

def test_despiking_robust_no_spikes():
    # Arrange: flat signal with no spikes
    signal = np.ones(100)

    # Act:
    result, _ = pre_processing.despiking_robust(
        signal,
        c=2.0,
        window_length=5
    )

    # Assert: unchanged signal
    np.testing.assert_array_equal(result, signal)


def test_despiking_robust_regular_case():
    # Arrange: 1 Hz signal, 1 hour duration => 3600 points
    size = 1 * 60 * 60 * 1
    spike_indices = [900, 1800, 2700]  # 15, 30, 45 min
    spike_value = 200  # well above threshold

    # Arrange: generate data with known spikes
    array = generate_uniform_data_with_spikes(size,
                                              spike_indices,
                                              spike_value)

    c = 3
    window_length = 5 * 60 + 1

    # Act:
    despiked_array, count_spike = pre_processing.despiking_robust(
        array,
        c=c,
        window_length=window_length
    )

    # Arrange: compute reference median values
    running_median, _ = core.running_stats_robust(array, window_length)

    # Assert: spikes were replaced with median values
    for idx in spike_indices:
        assert despiked_array[idx] == running_median[idx], (
            f"Spike at index {idx} was not removed: value {despiked_array[idx]}"
        )

    # Assert: correct number of spikes identified
    assert count_spike == len(spike_indices), (
        f"Expected {len(spike_indices)} spikes, but got {count_spike}"
    )


def test_despiking_robust_preserves_normal_values():
    # Arrange: normal-distributed signal with no spikes
    # Arrange: 1 Hz signal, duration: 1 hour => 3600 points
    size = 1 * 60 * 60 * 1
    # Arrange: no spikes in the input array
    array = generate_normal_data_with_spikes(size, 
                                             [], 
                                             0)

    c = 5.0
    window_length = 5

    # Act:
    despiked_array, _ = pre_processing.despiking_robust(
        array,
        c=c,
        window_length=window_length
    )

    # Assert: values preserved (within float equality)
    np.testing.assert_array_equal(
        array,
        despiked_array,
        err_msg="Non-spike values incorrectly modified."
    )

def test_despiking_robust_errors():
    # Arrange: valid input array
    input_array = np.ones(10)

    # Act & Assert: invalid c = 0
    with pytest.raises(ValueError, match="positive"):
        pre_processing.despiking_robust(input_array, c=0, window_length=3)

    # Act & Assert: invalid c < 0
    with pytest.raises(ValueError, match="positive"):
        pre_processing.despiking_robust(input_array, c=-1.0, window_length=3)

    # Act & Assert: invalid window_length = 0
    with pytest.raises(ValueError, match="positive"):
        pre_processing.despiking_robust(input_array, c=2.0, window_length=0)

    # Act & Assert: invalid window_length < 0
    with pytest.raises(ValueError, match="positive"):
        pre_processing.despiking_robust(input_array, c=2.0, window_length=-5)

    # Act & Assert: non-integer window_length
    with pytest.raises(ValueError, match="positive"):
        pre_processing.despiking_robust(input_array, c=2.0, window_length=4.5)


#######################################################################
################ testing pre_processing.interp_nan() ##################
#######################################################################

def test_interp_nan_no_nans():
    # Arrange: input array with no NaNs
    array = np.array([1.0, 2.0, 3.0])

    # Act:
    result, count = pre_processing.interp_nan(array)

    # Assert: no changes, no NaNs
    np.testing.assert_array_equal(result, array)
    assert count == 0


def test_interp_nan_single_nan():
    # Arrange: input array with a single NaN value
    array = np.array([1.0, np.nan, 3.0])
    expected = np.array([1.0, 2.0, 3.0])

    # Act:
    result, count = pre_processing.interp_nan(array)

    # Assert: NaN interpolated correctly
    np.testing.assert_allclose(result, expected)
    assert count == 1


def test_interp_nan_multiple_nans():
    # Arrange: input array with multiple consecutive NaN values
    array = np.array([1.0, np.nan, np.nan, 4.0])
    expected = np.array([1.0, 2.0, 3.0, 4.0])

    # Act:
    result, count = pre_processing.interp_nan(array)

    # Assert: NaNs interpolated correctly
    np.testing.assert_allclose(result, expected)
    assert count == 2


def test_interp_nan_nan_at_edges():
    # Arrange: input array with NaNs at the edges
    array = np.array([np.nan, 1.0, 2.0, np.nan])
    expected = np.array([np.nan, 1.0, 2.0, np.nan])

    # Act:
    result, count = pre_processing.interp_nan(array)

    # Assert: no interpolation at edges (NaNs remain)
    np.testing.assert_array_equal(result, expected)
    assert count == 0


def test_interp_nan_all_nans():
    # Arrange: input array with all NaN values
    array = np.array([np.nan, np.nan])
    expected = np.array([np.nan, np.nan])

    # Act:
    result, count = pre_processing.interp_nan(array)

    # Assert: all NaNs remain (cannot interpolate all NaNs)
    np.testing.assert_array_equal(result, expected)
    assert count == 0

#######################################################################
######### testing pre_processing.rotation_to_LEC_reference() ##########
#######################################################################

def test_rotation_to_LEC_reference_shape_mismatch_error():
    # Arrange: input with wrong shape for wind (2 instead of 3)
    wind = np.array([
        [1, 2, 3, 4],
        [0, 0, 0, 0],
    ])
    azimuth = 0
    model = "RM_YOUNG_81000"

    # Act & Assert: check that ValueError is raised
    with pytest.raises(ValueError, match="must have shape"):
        pre_processing.rotation_to_LEC_reference(wind, azimuth, model)


def test_rotation_to_LEC_reference_invalid_azimuth():
    # Arrange: wind array with valid shape
    wind = np.zeros((3, 10))

    # Act & Assert: azimuth < 0
    # check that ValueError is raised
    azimuth = -10
    model = "RM_YOUNG_81000"
    with pytest.raises(ValueError, match="azimuth is outside the range"):
        pre_processing.rotation_to_LEC_reference(wind, azimuth, model)

    # Act & Assert: azimuth > 360
    # check that ValueError is raised
    azimuth = 400
    model = "CAMPBELL_CSAT3"
    with pytest.raises(ValueError, match="azimuth is outside the range"):
        pre_processing.rotation_to_LEC_reference(wind, azimuth, model)

def test_rotation_to_LEC_reference_invalid_model():
    # Arrange: valid wind array and azimuth, invalid model
    wind = np.zeros((3, 10))
    azimuth = 0
    model = "BEST_ANEMOMETER_EVER"

    # Act & Assert: check that ValueError is raised for unknown model
    with pytest.raises(ValueError, match="Unknown model"):
        pre_processing.rotation_to_LEC_reference(wind, azimuth, model)


def test_rotation_to_LEC_reference_RM_YOUNG_81000_azimuth_zero():
    # Arrange: input wind and azimuth
    wind = np.array([
        [2, -2, 2, 2],
        [3, 3, -3, 3],
        [4, 4, 4, -4]
    ])
    azimuth = 0
    model = "RM_YOUNG_81000"
    
    # Act: perform rotation
    result = pre_processing.rotation_to_LEC_reference(wind, azimuth, model)

    # Assert: expected result after rotation
    expected = np.array([
        -wind[0,:],  # x component inverted
        -wind[1,:],  # y component inverted
         wind[2,:]   # z component unchanged
    ])
    np.testing.assert_array_equal(result, expected, err_msg="Something wrong...")

def test_rotation_to_LEC_reference_CAMPBELL_CSAT3_azimuth_zero():
    # Arrange: input wind and azimuth
    wind = np.array([
        [2, -2, 2, 2],
        [3, 3, -3, 3],
        [4, 4, 4, -4]
    ])
    azimuth = 0
    model = "CAMPBELL_CSAT3"
    
    # Act: perform rotation
    result = pre_processing.rotation_to_LEC_reference(wind, azimuth, model)
    
    # Assert: expected result after rotation
    expected = np.array([
         wind[1,:],   # new x = old y
        -wind[0,:],   # new y =-old x
         wind[2,:]    # z component unchanged
    ])
    
    np.testing.assert_array_equal(result, expected, err_msg="Something wrong...")

def test_rotation_to_LEC_reference_RM_YOUNG_81000_with_azimuth():
    # Arrange: input wind and azimuth
    wind = np.array([
        [2, -2, 2, 2],
        [3, 3, -3, 3],
        [4, 4, 4, -4]
    ])
    azimuth = 43
    model = "RM_YOUNG_81000"
    
    # Act: perform rotation with azimuth
    result = pre_processing.rotation_to_LEC_reference(wind, azimuth, model)

    # Rotation matrix for azimuth
    azimuth_rad = np.deg2rad(azimuth)
    rot_azimuth = np.zeros((3,3))
    rot_azimuth[0, 0] =  np.cos(azimuth_rad)
    rot_azimuth[0, 1] =  np.sin(azimuth_rad)
    rot_azimuth[1, 0] = -np.sin(azimuth_rad)
    rot_azimuth[1, 1] =  np.cos(azimuth_rad)
    rot_azimuth[2, 2] =  1

    wind_LEC = np.array([
        -wind[0,:],  # x component inverted
        -wind[1,:],  # y component inverted
         wind[2,:]   # z component unchanged
    ])
    expected = rot_azimuth @ wind_LEC
    
    # Assert: the result is close to the expected
    np.testing.assert_almost_equal(result, expected, decimal=4, err_msg="Something wrong...")

def test_rotation_to_LEC_reference_CAMPBELL_CSAT3_with_azimuth():
    # Arrange: input wind and azimuth
    wind = np.array([
        [2, -2, 2, 2],
        [3, 3, -3, 3],
        [4, 4, 4, -4]
    ])
    azimuth = 276
    model = "CAMPBELL_CSAT3"
    
    # Act: perform rotation with azimuth
    result = pre_processing.rotation_to_LEC_reference(wind, azimuth, model)
    
    # Rotation matrix for azimuth
    azimuth_rad = np.deg2rad(azimuth)
    rot_azimuth = np.zeros((3,3))
    rot_azimuth[0, 0] =  np.cos(azimuth_rad)
    rot_azimuth[0, 1] =  np.sin(azimuth_rad)
    rot_azimuth[1, 0] = -np.sin(azimuth_rad)
    rot_azimuth[1, 1] =  np.cos(azimuth_rad)
    rot_azimuth[2, 2] =  1

    wind_LEC = np.array([
         wind[1,:],   # new x = old y
        -wind[0,:],   # new y = -old x
         wind[2,:]    # z component unchanged
    ])
    expected = rot_azimuth @ wind_LEC
    
    # Assert: the result is close to the expected
    np.testing.assert_array_equal(result, expected, err_msg="Something wrong...")

#######################################################################
###### testing pre_processing.rotation_to_streamline_reference() ######
#######################################################################

def test_rotation_to_streamline_reference_no_rotation():
    # Arrange: Wind aligned with the x-axis, no rotation expected
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

    # Act: Perform rotation (no rotation expected)
    wind_rotated = pre_processing.rotation_to_streamline_reference(wind, wind_averaged)

    # Assert: The result should match the original wind array
    np.testing.assert_allclose(wind_rotated, wind, atol=1e-8)


def test_rotation_to_streamline_reference_shape_mismatch_error():
    # Arrange: Wind with incorrect shape (wrong first dimension)
    wind = np.array([
        [1, 2, 3, 4],
        [0, 0, 0, 0],
    ])
    wind_averaged = np.array([
        [1, 2, 3, 4],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])

    # Act & Assert: Check that ValueError is raised for incorrect shape
    with pytest.raises(ValueError, match="must have shape"):
        pre_processing.rotation_to_streamline_reference(wind, wind_averaged)

def test_rotation_to_streamline_reference_column_mismatch_error():
    # Arrange: Wind with 4 columns, wind_averaged with 5 columns
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

    # Act & Assert: Check that ValueError is raised for column mismatch
    with pytest.raises(ValueError, match="same number of columns"):
        pre_processing.rotation_to_streamline_reference(wind, wind_averaged)

def test_rotation_to_streamline_reference_output_shape():
    # Arrange: Wind array and averaged wind array
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

    # Act: Perform rotation
    wind_rotated = pre_processing.rotation_to_streamline_reference(wind, wind_averaged)

    # Assert: Check that the output has the expected shape
    assert wind_rotated.shape == (3, 4)

def test_rotation_to_streamline_reference_wind_along_y():
    # Arrange: Wind purely along the y-axis: (0, v, 0) for each sample
    wind = np.array([
        [0, 0, 0, 0],  # u component (no wind in the x direction)
        [1, 2, 3, 4],  # v component (wind aligned along the y-axis)
        [0, 0, 0, 0],  # w component (no vertical wind)
    ])

    # Averaged wind is purely horizontal along y (same for each sample)
    wind_averaged = np.array([
        [0, 0, 0, 0],  # mean u component (no horizontal wind in x)
        [1, 2, 3, 4],  # mean v component (aligned along y-axis)
        [0, 0, 0, 0],  # mean w component (no vertical wind)
    ])

    # Act: Perform rotation to streamline reference
    wind_rotated = pre_processing.rotation_to_streamline_reference(wind, wind_averaged)

    # Assert: After rotation, the x component should match the v component in magnitude
    np.testing.assert_allclose(wind_rotated[0, :], wind_averaged[1, :], atol=1e-8)
    
    # The y and z components should be zero
    np.testing.assert_allclose(wind_rotated[1, :], 0, atol=1e-8)
    np.testing.assert_allclose(wind_rotated[2, :], 0, atol=1e-8)

def test_rotation_to_streamline_reference_mean_v_nullified():
    # Build a fake wind signal at 1 Hz
    # Every 10 minutes the wind changes direction => each segment in which the wind is constant has length 10*60 points
    # From North, from N-E, from E, etc. with angle differences of 45 degrees between segments
    # 8 directions => 8*10min = 8*10*60 points
    
    # Define base wind components (u, v, w) for the 8 wind directions (N, NE, E, SE, S, SW, W, NW)
    u_base = [ 0, -1/np.sqrt(2), -1, -1/np.sqrt(2), 0,
            1/np.sqrt(2), 1,  1/np.sqrt(2)]
    v_base = [-1, -1/np.sqrt(2),  0,  1/np.sqrt(2), 1, 
            1/np.sqrt(2), 0, -1/np.sqrt(2)]
    w_base = [0, 0, 0, 0, 0,
                0, 0, 0]
    
    # Length of each segment (15 minutes)
    length_single = 15 * 60

    # Create the wind signal by repeating the base wind components for each segment
    wind = np.array([np.repeat(u_base, length_single),
                        np.repeat(v_base, length_single),
                        np.repeat(w_base, length_single)])

    # Initialize the averaged wind signal with zeros
    wind_averaged = np.full(wind.shape, 0.0)
    
    # Define the window length for averaging (5 minutes + 1 for odd number of points)
    window_length = 5*60 + 1  # 5 minute window, odd number of points
    
    # Calculate the running averages for each component (u, v, w)
    for i in range(3):
        wind_averaged[i,:], _ = core.running_stats(wind[i,:], window_length)

    # Rotate the wind signal to the streamline reference using the averaged wind
    wind_rotated_result = pre_processing.rotation_to_streamline_reference(wind,
                                                                            wind_averaged)

    # Calculate the running average of the rotated v component (y direction)
    v_rotated_result_averaged, _ = core.running_stats(wind_rotated_result[1,:],
                                                    window_length)

    # Define the minutes where we want to check the results (5 minutes intervals)
    start_minutes = [5, 20, 35, 50, 65, 80, 95, 110]  # start minutes

    # Create an index list to control the points for each 5-minute interval
    index_list_to_control = []
    for start_min in start_minutes:
        start_idx = start_min * 60
        end_idx = (start_min + 5) * 60  # 5 minutes later
        index_list_to_control.append(np.arange(start_idx, end_idx + 1, 1))

    # Define the threshold to consider as significant
    threshold = 1e-5  # or any other value you prefer

    # Flatten all indices to check into a single array
    indices_to_check = np.concatenate(index_list_to_control)

    # Create a mask for the points that exceed the threshold
    mask_over_threshold = np.abs(v_rotated_result_averaged[indices_to_check]) > threshold

    # Find the actual indices of points above the threshold
    bad_indices = indices_to_check[mask_over_threshold]
    bad_values = v_rotated_result_averaged[bad_indices]

    # Comment explaining the rationale:
    # I want to check that the running average of the wind along the y-direction in the rotated series is zero 
    # at the time points where the initial (non-rotated) wind is sufficiently constant. This check is done 
    # between the 5th and 10th minute after the wind direction change, when the wind is stable.

    # If any points exceed the threshold, raise an error with details
    if bad_indices.size > 0:
        error_message = (
            f"Found {bad_indices.size} points exceeding the threshold {threshold}.\n"
            f"indices = {bad_indices.tolist()}\n"
            f"values = {bad_values.tolist()}"
        )
        np.testing.assert_equal(
            bad_indices.size, 0,
            err_msg=error_message
        )

#######################################################################
########### testing pre_processing.wind_dir_LEC_reference() ###########
#######################################################################

def test_wind_dir_LEC_reference_regular_case_scalar():
    # Arrange: define the u and v components of the wind vector
    u = 10
    v = 10
    
    # Act: compute the wind direction
    result = pre_processing.wind_dir_LEC_reference(u, v)
    expected = 225  # 270-45 or 180+45, wind from S-W
    
    # Assert: wind direction is correct
    assert np.isclose(result, expected, rtol=1e-5)

def test_wind_dir_LEC_reference_regular_case_array():
    # Arrange: define u and v components of wind vectors and the expected wind directions
    u = [ 0, -1/np.sqrt(2), -1, -1/np.sqrt(2), 0,
          1/np.sqrt(2), 1,  1/np.sqrt(2)]
    v = [-1, -1/np.sqrt(2),  0,  1/np.sqrt(2), 1, 
          1/np.sqrt(2), 0, -1/np.sqrt(2)]
    wind_dir_expected = [0, 45, 90, 135, 180, 225, 270, 315]

    # Act: compute the wind directions
    result = pre_processing.wind_dir_LEC_reference(u, v)
    
    # Assert: the computed wind directions match the expected values
    np.testing.assert_allclose(result, wind_dir_expected, rtol=1e-5)


def test_wind_dir_LEC_reference_shape_mismatch():
    # Arrange: define u and v with mismatched shapes
    u = np.array([1, 2, 3])
    v = np.array([1, 2])
    
    # Act & Assert: expect a ValueError due to shape mismatch
    with pytest.raises(ValueError, match="Shape mismatch"):
        pre_processing.wind_dir_LEC_reference(u, v)

def test_wind_dir_LEC_reference_negative_threshold():
    # Arrange: define u, v, and a negative threshold
    u = [0]
    v = [1]
    threshold = -2

    # Act & Assert: expect a ValueError due to the negative threshold
    with pytest.raises(ValueError, match="positive"):
        pre_processing.wind_dir_LEC_reference(u, v, threshold)

def test_wind_dir_LEC_reference_threshold():
    # Arrange: define u, v and threshold for low wind speeds
    u = np.array([0.01, 1.0])
    v = np.array([0.01, 0.0])
    threshold = 0.1

    # Act: compute the wind direction considering the threshold
    result = pre_processing.wind_dir_LEC_reference(u, v, threshold=threshold)

    # Assert: check that low wind speed results in NaN, and that the second result is close to 270 degrees
    assert np.isnan(result[0]), f"Expected NaN for low wind speed, got {result[0]}"
    assert np.isclose(result[1], 270.0, atol=1e-2), f"Expected ~90 degrees, got {result[1]}"

#######################################################################
##### testing pre_processing.wind_dir_modeldependent_reference() ######
#######################################################################

def test_wind_dir_modeldependent_reference_scalar():
    # Arrange: define u, v components of wind and azimuth angle
    u = 10
    v = 10
    azimuth = 0.0

    # Act & Assert: Test for RM_YOUNG_81000 model
    result_rm = pre_processing.wind_dir_modeldependent_reference(u, v, azimuth, model="RM_YOUNG_81000")
    expected_rm = 45  # inverts u and v, so from NE
    assert np.isclose(result_rm, expected_rm, rtol=1e-5)

    # Act & Assert: Test for CAMPBELL_CSAT3 model
    result_cs = pre_processing.wind_dir_modeldependent_reference(u, v, azimuth, model="CAMPBELL_CSAT3")
    expected_cs = 315.0  # because v_LEC = -10, u_LEC = 10 => (-10, 10) -> 315°
    assert np.isclose(result_cs, expected_cs, rtol=1e-5)

def test_wind_dir_modeldependent_reference_shape_mismatch():
    # Arrange: define u and v with mismatched shapes
    u = np.array([1, 2, 3])
    v = np.array([1, 2])
    azimuth = 0.0

    # Act & Assert: expect a ValueError due to shape mismatch
    with pytest.raises(ValueError, match="Shape mismatch"):
        pre_processing.wind_dir_modeldependent_reference(u, v, azimuth, model="RM_YOUNG_81000")

def test_wind_dir_modeldependent_reference_unknown_model():
    # Arrange: define u, v components and azimuth
    u = [0]
    v = [1]
    azimuth = 0.0

    # Act & Assert: expect a ValueError due to unknown model
    with pytest.raises(ValueError, match="Unknown model"):
        pre_processing.wind_dir_modeldependent_reference(u, v, azimuth, model="UNKNOWN_MODEL")

def test_wind_dir_modeldependent_reference_negative_threshold():
    # Arrange: define u, v components, azimuth, and a negative threshold
    u = [0]
    v = [1]
    azimuth = 0.0
    model = "RM_YOUNG_81000"
    threshold = -2

    # Act & Assert: expect a ValueError due to the negative threshold
    with pytest.raises(ValueError, match="positive"):
        pre_processing.wind_dir_modeldependent_reference(u, v, azimuth, model, threshold)

def test_wind_dir_modeldependent_reference_no_azimuth():
    # Arrange: define u, v components of wind and azimuth
    u = [ 0, -1/np.sqrt(2), -1, -1/np.sqrt(2), 0,
          1/np.sqrt(2), 1,  1/np.sqrt(2)]
    v = [-1, -1/np.sqrt(2),  0,  1/np.sqrt(2), 1, 
          1/np.sqrt(2), 0, -1/np.sqrt(2)]
    azimuth = 0.0

    # Act: Test for RM_YOUNG_81000 model with no azimuth
    result_rm = pre_processing.wind_dir_modeldependent_reference(u, v, azimuth, model="RM_YOUNG_81000")
    expected_rm = [(angle + 180) % 360 for angle in [0, 45, 90, 135, 180, 225, 270, 315]]
    np.testing.assert_allclose(result_rm, expected_rm, rtol=1e-5)

    # Act: Test for CAMPBELL_CSAT3 model with no azimuth
    result_cs = pre_processing.wind_dir_modeldependent_reference(u, v, azimuth, model="CAMPBELL_CSAT3")
    expected_cs = [(angle + 90) % 360 for angle in [0, 45, 90, 135, 180, 225, 270, 315]]
    np.testing.assert_allclose(result_cs, expected_cs, rtol=1e-5)

def test_wind_dir_modeldependent_reference_with_azimuth():
    # Arrange: define u, v components of wind and azimuth with a 30° rotation
    u = [0, -1/np.sqrt(2), -1, -1/np.sqrt(2), 0,
         1/np.sqrt(2), 1, 1/np.sqrt(2)]
    v = [-1, -1/np.sqrt(2), 0, 1/np.sqrt(2), 1, 
         1/np.sqrt(2), 0, -1/np.sqrt(2)]
    azimuth = 30.0  # instrument rotated by 30°

    # Act: Test for RM_YOUNG_81000 model with azimuth
    result_rm = pre_processing.wind_dir_modeldependent_reference(u, v, azimuth, model="RM_YOUNG_81000")
    expected_rm = [((angle + 180) - azimuth) % 360 for angle in [0, 45, 90, 135, 180, 225, 270, 315]]
    np.testing.assert_allclose(result_rm, expected_rm, rtol=1e-5)

    # Act: Test for CAMPBELL_CSAT3 model with azimuth
    result_cs = pre_processing.wind_dir_modeldependent_reference(u, v, azimuth, model="CAMPBELL_CSAT3")
    expected_cs = [((angle + 90) - azimuth) % 360 for angle in [0, 45, 90, 135, 180, 225, 270, 315]]
    np.testing.assert_allclose(result_cs, expected_cs, rtol=1e-5)

def test_wind_dir_modeldependent_reference_threshold():
    # Arrange: define u, v components, threshold, and azimuth
    u = np.array([0.01, 1.0])
    v = np.array([0.01, 0.0])
    threshold = 0.1
    azimuth = 0.0

    # Act & Assert: Test for RM_YOUNG_81000 model with threshold
    result = pre_processing.wind_dir_modeldependent_reference(u, v, azimuth, model="RM_YOUNG_81000", threshold=threshold)
    assert np.isnan(result[0]), f"Expected NaN for low wind speed, got {result[0]}"
    assert np.isclose(result[1], 90.0, atol=1e-2), f"Expected ~270 degrees, got {result[1]}"

    # Act & Assert: Test for CAMPBELL_CSAT3 model with threshold
    result2 = pre_processing.wind_dir_modeldependent_reference(u, v, azimuth, model="CAMPBELL_CSAT3", threshold=threshold)
    assert np.isnan(result2[0]), f"Expected NaN for low wind speed, got {result2[0]}"
    assert np.isclose(result2[1], 0.0, atol=1e-2), f"Expected ~0 degrees, got {result2[1]}"

def test_comparison_wind_dir_methods_LEC_modeldependent_noazimuth():
    # Arrange: Define wind components (u, v), w (constant), and azimuth for RM_YOUNG_81000 model
    u = [0, -1/np.sqrt(2), -1, -1/np.sqrt(2), 0,
         1/np.sqrt(2), 1, 1/np.sqrt(2)]
    v = [-1, -1/np.sqrt(2), 0, 1/np.sqrt(2), 1, 
         1/np.sqrt(2), 0, -1/np.sqrt(2)]
    w = np.full(len(u), 0)
    
    azimuth = 0
    wind = np.array([u, v, w])
    model = "RM_YOUNG_81000"

    # Act: Compute wind direction using modeldependent method
    wind_dir_result_modeldependent = pre_processing.wind_dir_modeldependent_reference(u, v, azimuth, model)

    # Act: Compute wind direction using LEC method
    wind_LEC = pre_processing.rotation_to_LEC_reference(wind, azimuth, model)
    wind_dir_result_LEC = pre_processing.wind_dir_LEC_reference(wind_LEC[0, :], wind_LEC[1, :])
    
    # Assert: Check if the results from the two methods are equal
    np.testing.assert_array_equal(wind_dir_result_modeldependent, wind_dir_result_LEC, 
                                  f"For model {model}: wind directions computed with the two different methods do NOT match!")

    # Repeat for CAMPBELL_CSAT3 model
    model = "CAMPBELL_CSAT3"

    # Act: Compute wind direction using modeldependent method
    wind_dir_result_modeldependent = pre_processing.wind_dir_modeldependent_reference(u, v, azimuth, model)

    # Act: Compute wind direction using LEC method
    wind_LEC = pre_processing.rotation_to_LEC_reference(wind, azimuth, model)
    wind_dir_result_LEC = pre_processing.wind_dir_LEC_reference(wind_LEC[0, :], wind_LEC[1, :])
    
    # Assert: Check if the results from the two methods are equal
    np.testing.assert_array_equal(wind_dir_result_modeldependent, wind_dir_result_LEC, 
                                  f"For model {model}: wind directions computed with the two different methods do NOT match!")


def test_comparison_wind_dir_methods_LEC_modeldependent_with_azimuth():
    # Arrange: Define wind components (u, v), w (constant), and azimuth for RM_YOUNG_81000 model
    u = [0, -1/np.sqrt(2), -1, -1/np.sqrt(2), 0,
         1/np.sqrt(2), 1, 1/np.sqrt(2)]
    v = [-1, -1/np.sqrt(2), 0, 1/np.sqrt(2), 1, 
         1/np.sqrt(2), 0, -1/np.sqrt(2)]
    w = np.full(len(u), 0)
    
    azimuth = 31
    wind = np.array([u, v, w])
    model = "RM_YOUNG_81000"

    # Act: Compute wind direction using modeldependent method
    wind_dir_result_modeldependent = pre_processing.wind_dir_modeldependent_reference(u, v, azimuth, model)

    # Act: Compute wind direction using LEC method
    wind_LEC = pre_processing.rotation_to_LEC_reference(wind, azimuth, model)
    wind_dir_result_LEC = pre_processing.wind_dir_LEC_reference(wind_LEC[0, :], wind_LEC[1, :])
    
    # Assert: Check if the results from the two methods are equal
    np.testing.assert_array_equal(wind_dir_result_modeldependent, wind_dir_result_LEC, 
                                  f"For model {model}: wind directions computed with the two different methods do NOT match!")

    # Repeat for CAMPBELL_CSAT3 model
    model = "CAMPBELL_CSAT3"

    # Act: Compute wind direction using modeldependent method
    wind_dir_result_modeldependent = pre_processing.wind_dir_modeldependent_reference(u, v, azimuth, model)

    # Act: Compute wind direction using LEC method
    wind_LEC = pre_processing.rotation_to_LEC_reference(wind, azimuth, model)
    wind_dir_result_LEC = pre_processing.wind_dir_LEC_reference(wind_LEC[0, :], wind_LEC[1, :])
    
    # Assert: Check if the results from the two methods are equal
    np.testing.assert_array_equal(wind_dir_result_modeldependent, wind_dir_result_LEC, 
                                  f"For model {model}: wind directions computed with the two different methods do NOT match!")

#######################################################################
#######################################################################
#######################################################################