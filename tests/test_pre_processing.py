import sys
import os
import pytest
import pandas as pd
import numpy as np
import logging
from unittest.mock import MagicMock
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
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