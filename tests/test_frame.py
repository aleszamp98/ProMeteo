import pytest
import pandas as pd
import numpy as np
import logging
from unittest.mock import MagicMock
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import frame
import core

#######################################################################
######### testing frame.rotation_to_LEC_reference() ##########
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
        frame.rotation_to_LEC_reference(wind, azimuth, model)


def test_rotation_to_LEC_reference_invalid_azimuth():
    # Arrange: wind array with valid shape
    wind = np.zeros((3, 10))

    # Act & Assert: azimuth < 0
    # check that ValueError is raised
    azimuth = -10
    model = "RM_YOUNG_81000"
    with pytest.raises(ValueError, match="azimuth is outside the range"):
        frame.rotation_to_LEC_reference(wind, azimuth, model)

    # Act & Assert: azimuth > 360
    # check that ValueError is raised
    azimuth = 400
    model = "CAMPBELL_CSAT3"
    with pytest.raises(ValueError, match="azimuth is outside the range"):
        frame.rotation_to_LEC_reference(wind, azimuth, model)

def test_rotation_to_LEC_reference_invalid_model():
    # Arrange: valid wind array and azimuth, invalid model
    wind = np.zeros((3, 10))
    azimuth = 0
    model = "BEST_ANEMOMETER_EVER"

    # Act & Assert: check that ValueError is raised for unknown model
    with pytest.raises(ValueError, match="Unknown model"):
        frame.rotation_to_LEC_reference(wind, azimuth, model)


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
    result = frame.rotation_to_LEC_reference(wind, azimuth, model)

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
    result = frame.rotation_to_LEC_reference(wind, azimuth, model)
    
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
    result = frame.rotation_to_LEC_reference(wind, azimuth, model)

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
    result = frame.rotation_to_LEC_reference(wind, azimuth, model)
    
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
###### testing frame.rotation_to_streamline_reference() ######
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
    wind_rotated = frame.rotation_to_streamline_reference(wind, wind_averaged)

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
        frame.rotation_to_streamline_reference(wind, wind_averaged)

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
        frame.rotation_to_streamline_reference(wind, wind_averaged)

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
    wind_rotated = frame.rotation_to_streamline_reference(wind, wind_averaged)

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
    wind_rotated = frame.rotation_to_streamline_reference(wind, wind_averaged)

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
    wind_rotated_result = frame.rotation_to_streamline_reference(wind,
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
########### testing frame.wind_dir_LEC_reference() ###########
#######################################################################

def test_wind_dir_LEC_reference_regular_case_scalar():
    # Arrange: define the u and v components of the wind vector
    u = 10
    v = 10
    
    # Act: compute the wind direction
    result = frame.wind_dir_LEC_reference(u, v)
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
    result = frame.wind_dir_LEC_reference(u, v)
    
    # Assert: the computed wind directions match the expected values
    np.testing.assert_allclose(result, wind_dir_expected, rtol=1e-5)


def test_wind_dir_LEC_reference_shape_mismatch():
    # Arrange: define u and v with mismatched shapes
    u = np.array([1, 2, 3])
    v = np.array([1, 2])
    
    # Act & Assert: expect a ValueError due to shape mismatch
    with pytest.raises(ValueError, match="Shape mismatch"):
        frame.wind_dir_LEC_reference(u, v)

def test_wind_dir_LEC_reference_negative_threshold():
    # Arrange: define u, v, and a negative threshold
    u = [0]
    v = [1]
    threshold = -2

    # Act & Assert: expect a ValueError due to the negative threshold
    with pytest.raises(ValueError, match="positive"):
        frame.wind_dir_LEC_reference(u, v, threshold)

def test_wind_dir_LEC_reference_threshold():
    # Arrange: define u, v and threshold for low wind speeds
    u = np.array([0.01, 1.0])
    v = np.array([0.01, 0.0])
    threshold = 0.1

    # Act: compute the wind direction considering the threshold
    result = frame.wind_dir_LEC_reference(u, v, threshold=threshold)

    # Assert: check that low wind speed results in NaN, and that the second result is close to 270 degrees
    assert np.isnan(result[0]), f"Expected NaN for low wind speed, got {result[0]}"
    assert np.isclose(result[1], 270.0, atol=1e-2), f"Expected ~90 degrees, got {result[1]}"

#######################################################################
##### testing frame.wind_dir_modeldependent_reference() ######
#######################################################################

def test_wind_dir_modeldependent_reference_scalar():
    # Arrange: define u, v components of wind and azimuth angle
    u = 10
    v = 10
    azimuth = 0.0

    # Act & Assert: Test for RM_YOUNG_81000 model
    result_rm = frame.wind_dir_modeldependent_reference(u, v, azimuth, model="RM_YOUNG_81000")
    expected_rm = 45  # inverts u and v, so from NE
    assert np.isclose(result_rm, expected_rm, rtol=1e-5)

    # Act & Assert: Test for CAMPBELL_CSAT3 model
    result_cs = frame.wind_dir_modeldependent_reference(u, v, azimuth, model="CAMPBELL_CSAT3")
    expected_cs = 315.0  # because v_LEC = -10, u_LEC = 10 => (-10, 10) -> 315°
    assert np.isclose(result_cs, expected_cs, rtol=1e-5)

def test_wind_dir_modeldependent_reference_shape_mismatch():
    # Arrange: define u and v with mismatched shapes
    u = np.array([1, 2, 3])
    v = np.array([1, 2])
    azimuth = 0.0

    # Act & Assert: expect a ValueError due to shape mismatch
    with pytest.raises(ValueError, match="Shape mismatch"):
        frame.wind_dir_modeldependent_reference(u, v, azimuth, model="RM_YOUNG_81000")

def test_wind_dir_modeldependent_reference_unknown_model():
    # Arrange: define u, v components and azimuth
    u = [0]
    v = [1]
    azimuth = 0.0

    # Act & Assert: expect a ValueError due to unknown model
    with pytest.raises(ValueError, match="Unknown model"):
        frame.wind_dir_modeldependent_reference(u, v, azimuth, model="UNKNOWN_MODEL")

def test_wind_dir_modeldependent_reference_negative_threshold():
    # Arrange: define u, v components, azimuth, and a negative threshold
    u = [0]
    v = [1]
    azimuth = 0.0
    model = "RM_YOUNG_81000"
    threshold = -2

    # Act & Assert: expect a ValueError due to the negative threshold
    with pytest.raises(ValueError, match="positive"):
        frame.wind_dir_modeldependent_reference(u, v, azimuth, model, threshold)

def test_wind_dir_modeldependent_reference_no_azimuth():
    # Arrange: define u_LEC, LEC components of wind and azimuth
    u_LEC = np.array([ 0, -1/np.sqrt(2), -1, -1/np.sqrt(2), 0,
          1/np.sqrt(2), 1,  1/np.sqrt(2)])
    v_LEC = np.array([-1, -1/np.sqrt(2),  0,  1/np.sqrt(2), 1, 
          1/np.sqrt(2), 0, -1/np.sqrt(2)])
    # Arrange: compute wind direction in the meteorological convention using wind_dir_LEC
    expected = frame.wind_dir_LEC_reference(u_LEC, v_LEC, threshold = 0)
    # Arrange: set null azimuth
    azimuth = 0.0

    # Arrange: compute the wind components as measured by RM YOUNG anemometer
    u_measured_YOUNG = - u_LEC
    v_measured_YOUNG = - v_LEC
    # Act: Test for RM_YOUNG_81000 model with no azimuth
    result_RM_YOUNG = frame.wind_dir_modeldependent_reference(u_measured_YOUNG,
                                                                       v_measured_YOUNG,
                                                                       azimuth,
                                                                       model="RM_YOUNG_81000")
    # Assert: result as expected
    np.testing.assert_allclose(result_RM_YOUNG, expected, rtol=1e-5)

    # Arrange: compute the wind components as measured by CAMPBELL CSAT3 anemometer
    u_measured_CSAT = - v_LEC
    v_measured_CSAT =   u_LEC
    # Act: Test for RM_YOUNG_81000 model with no azimuth
    result_CSAT = frame.wind_dir_modeldependent_reference(u_measured_CSAT,
                                                                   v_measured_CSAT,
                                                                   azimuth,
                                                                   model="CAMPBELL_CSAT3")
    # Assert: result as expected
    np.testing.assert_allclose(result_CSAT, expected, rtol=1e-5)

def test_wind_dir_modeldependent_reference_with_azimuth():
    # --- Case A: acute azimuth (30°) ---
    # Wind from:
    # - NE (40°)        => 1st quadrant
    # - SE (180-40=140) => 2nd quadrant
    # - SW (180+40=220) => 3rd quadrant
    # - NW (270+40=310) => 4th quadrant
    azimuth_A = 30

    # Arrange: define wind components in a rotated frame of an angle `azimuth` with respect to the North
    u_measured_LEC_rot_A = np.array([
        -np.sin(np.deg2rad(40 - 30)),
        -np.cos(np.deg2rad(90 - 40 - 30)),
        np.sin(np.deg2rad(40 - 30)),
        np.cos(np.deg2rad(40 - 30))
    ])
    v_measured_LEC_rot_A = np.array([
        -np.cos(np.deg2rad(40 - 30)),
        np.sin(np.deg2rad(90 - 40 - 30)),
        np.cos(np.deg2rad(40 - 30)),
        -np.sin(np.deg2rad(40 - 30))
    ])
    wind_dir_expected_A = [40, 140, 220, 310]

    # Arrange: convert to RM YOUNG measured components
    u_measured_Y_A = -u_measured_LEC_rot_A
    v_measured_Y_A = -v_measured_LEC_rot_A

    # Act: compute wind direction with RM_YOUNG_81000 model
    result_RM_YOUNG_A = frame.wind_dir_modeldependent_reference(
        u_measured_Y_A,
        v_measured_Y_A,
        azimuth_A,
        model="RM_YOUNG_81000"
    )

    # Assert: compare with expected directions
    np.testing.assert_allclose(result_RM_YOUNG_A, wind_dir_expected_A, rtol=1e-5)

    # Arrange: convert to CSAT3 measured components
    u_measured_CSAT_A = -v_measured_LEC_rot_A
    v_measured_CSAT_A = u_measured_LEC_rot_A

    # Act: compute wind direction with CAMPBELL_CSAT3 model
    result_CSAT_A = frame.wind_dir_modeldependent_reference(
        u_measured_CSAT_A,
        v_measured_CSAT_A,
        azimuth_A,
        model="CAMPBELL_CSAT3"
    )

    # Assert: compare with expected directions
    np.testing.assert_allclose(result_CSAT_A, wind_dir_expected_A, rtol=1e-5)

    # --- Case B: obtuse azimuth (120°) ---
    # Wind from:
    # - NE (40°)        => 1st quadrant
    # - SE (180-40=140) => 2nd quadrant
    # - SW (180+40=220) => 3rd quadrant
    # - NW (270+40=310) => 4th quadrant
    azimuth_B = 120

    # Arrange: define wind components in LEC frame (rotated for 40° wind direction)
    u_measured_LEC_rot_B = np.array([
        np.cos(np.deg2rad(40 - 30)),
        -np.sin(np.deg2rad(20)),
        -np.cos(np.deg2rad(40 - 30)),
        np.sin(np.deg2rad(40 - 30))
    ])
    v_measured_LEC_rot_B = np.array([
        -np.sin(np.deg2rad(40 - 30)),
        -np.cos(np.deg2rad(20)),
        np.sin(np.deg2rad(40 - 30)),
        np.cos(np.deg2rad(40 - 30))
    ])
    wind_dir_expected_B = [40, 140, 220, 310]

    # Arrange: convert to RM YOUNG measured components
    u_measured_Y_B = -u_measured_LEC_rot_B
    v_measured_Y_B = -v_measured_LEC_rot_B

    # Act: compute wind direction with RM_YOUNG_81000 model
    result_RM_YOUNG_B = frame.wind_dir_modeldependent_reference(
        u_measured_Y_B,
        v_measured_Y_B,
        azimuth_B,
        model="RM_YOUNG_81000"
    )

    # Assert: compare with expected directions
    np.testing.assert_allclose(result_RM_YOUNG_B, wind_dir_expected_B, rtol=1e-5)

    # Arrange: convert to CSAT3 measured components
    u_measured_CSAT_B = -v_measured_LEC_rot_B
    v_measured_CSAT_B = u_measured_LEC_rot_B

    # Act: compute wind direction with CAMPBELL_CSAT3 model
    result_CSAT_B = frame.wind_dir_modeldependent_reference(
        u_measured_CSAT_B,
        v_measured_CSAT_B,
        azimuth_B,
        model="CAMPBELL_CSAT3"
    )

    # Assert: compare with expected directions
    np.testing.assert_allclose(result_CSAT_B, wind_dir_expected_B, rtol=1e-5)

    

def test_wind_dir_modeldependent_reference_threshold():
    # Arrange: define u, v components, threshold, and azimuth
    u = np.array([0.01, 1.0])
    v = np.array([0.01, 0.0])
    threshold = 0.1
    azimuth = 0.0

    # Act & Assert: Test for RM_YOUNG_81000 model with threshold
    result = frame.wind_dir_modeldependent_reference(u, v, azimuth, model="RM_YOUNG_81000", threshold=threshold)
    assert np.isnan(result[0]), f"Expected NaN for low wind speed, got {result[0]}"
    assert np.isclose(result[1], 90.0, atol=1e-2), f"Expected ~270 degrees, got {result[1]}"

    # Act & Assert: Test for CAMPBELL_CSAT3 model with threshold
    result2 = frame.wind_dir_modeldependent_reference(u, v, azimuth, model="CAMPBELL_CSAT3", threshold=threshold)
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
    wind_dir_result_modeldependent = frame.wind_dir_modeldependent_reference(u, v, azimuth, model)

    # Act: Compute wind direction using LEC method
    wind_LEC = frame.rotation_to_LEC_reference(wind, azimuth, model)
    wind_dir_result_LEC = frame.wind_dir_LEC_reference(wind_LEC[0, :], wind_LEC[1, :])
    
    # Assert: Check if the results from the two methods are equal
    np.testing.assert_array_equal(wind_dir_result_modeldependent, wind_dir_result_LEC, 
                                  f"For model {model}: wind directions computed with the two different methods do NOT match!")

    # Repeat for CAMPBELL_CSAT3 model
    model = "CAMPBELL_CSAT3"

    # Act: Compute wind direction using modeldependent method
    wind_dir_result_modeldependent = frame.wind_dir_modeldependent_reference(u, v, azimuth, model)

    # Act: Compute wind direction using LEC method
    wind_LEC = frame.rotation_to_LEC_reference(wind, azimuth, model)
    wind_dir_result_LEC = frame.wind_dir_LEC_reference(wind_LEC[0, :], wind_LEC[1, :])
    
    # Assert: Check if the results from the two methods are equal
    np.testing.assert_array_equal(wind_dir_result_modeldependent, wind_dir_result_LEC, 
                                  f"For model {model}: wind directions computed with the two different methods do NOT match!")


def test_comparison_wind_dir_methods_LEC_modeldependent_with_acute_azimuth():
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
    wind_dir_result_modeldependent = frame.wind_dir_modeldependent_reference(u, v, azimuth, model)

    # Act: Compute wind direction using LEC method
    wind_LEC = frame.rotation_to_LEC_reference(wind, azimuth, model)
    wind_dir_result_LEC = frame.wind_dir_LEC_reference(wind_LEC[0, :], wind_LEC[1, :])
    
    # Assert: Check if the results from the two methods are equal
    np.testing.assert_allclose(wind_dir_result_modeldependent, wind_dir_result_LEC, rtol=1e-12,
                                  err_msg=f"For model {model}: wind directions computed with the two different methods do NOT match!")

    # Repeat for CAMPBELL_CSAT3 model
    model = "CAMPBELL_CSAT3"

    # Act: Compute wind direction using modeldependent method
    wind_dir_result_modeldependent = frame.wind_dir_modeldependent_reference(u, v, azimuth, model)

    # Act: Compute wind direction using LEC method
    wind_LEC = frame.rotation_to_LEC_reference(wind, azimuth, model)
    wind_dir_result_LEC = frame.wind_dir_LEC_reference(wind_LEC[0, :], wind_LEC[1, :])
    
    # Assert: Check if the results from the two methods are equal
    np.testing.assert_allclose(wind_dir_result_modeldependent, wind_dir_result_LEC, rtol=1e-12, 
                                  err_msg=f"For model {model}: wind directions computed with the two different methods do NOT match!")
    
def test_comparison_wind_dir_methods_LEC_modeldependent_with_obtuse_azimuth():
    # Arrange: Define wind components (u, v), w (constant), and azimuth for RM_YOUNG_81000 model
    u = [0, -1/np.sqrt(2), -1, -1/np.sqrt(2), 0,
         1/np.sqrt(2), 1, 1/np.sqrt(2)]
    v = [-1, -1/np.sqrt(2), 0, 1/np.sqrt(2), 1, 
         1/np.sqrt(2), 0, -1/np.sqrt(2)]
    w = np.full(len(u), 0)
    
    azimuth = 270
    wind = np.array([u, v, w])
    model = "RM_YOUNG_81000"

    # Act: Compute wind direction using modeldependent method
    wind_dir_result_modeldependent = frame.wind_dir_modeldependent_reference(u, v, azimuth, model)

    # Act: Compute wind direction using LEC method
    wind_LEC = frame.rotation_to_LEC_reference(wind, azimuth, model)
    wind_dir_result_LEC = frame.wind_dir_LEC_reference(wind_LEC[0, :], wind_LEC[1, :])
    
    # Assert: Check if the results from the two methods are equal
    np.testing.assert_allclose(wind_dir_result_modeldependent, wind_dir_result_LEC, rtol=1e-12,
                                  err_msg=f"For model {model}: wind directions computed with the two different methods do NOT match!")

    # Repeat for CAMPBELL_CSAT3 model
    model = "CAMPBELL_CSAT3"

    # Act: Compute wind direction using modeldependent method
    wind_dir_result_modeldependent = frame.wind_dir_modeldependent_reference(u, v, azimuth, model)

    # Act: Compute wind direction using LEC method
    wind_LEC = frame.rotation_to_LEC_reference(wind, azimuth, model)
    wind_dir_result_LEC = frame.wind_dir_LEC_reference(wind_LEC[0, :], wind_LEC[1, :])
    
    # Assert: Check if the results from the two methods are equal
    np.testing.assert_allclose(wind_dir_result_modeldependent, wind_dir_result_LEC, rtol=1e-12, 
                                  err_msg=f"For model {model}: wind directions computed with the two different methods do NOT match!")

#######################################################################
#######################################################################
#######################################################################