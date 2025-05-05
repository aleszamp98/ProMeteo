import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, Union

def rotation_to_LEC_reference(wind : np.ndarray,
                              azimuth : float,
                              model : str
                              ) -> np.ndarray: 
    """
    Rotate the wind vector from the anemometer reference system 
    to the Local Earth Coordinate system (LEC), given the orientation `azimuth`
    of the anemometer head with respect to the North.

    Parameters
    ----------
    wind : np.ndarray
        A 3xN array (shape (3, N)), where each column is a wind vector at a different time instant.
        The three rows correspond to the three velocity components.
    azimuth : float
        The azimuth angle in degrees measured clockwise from the North, describing the orientation
        of the anemometer head with respect to the North-
    model : str
        The anemometer model used for the measurement. 
        Only two models are supported: "RM_YOUNG_81000", "CAMPBELL_CSAT3"

    Returns
    -------
    wind_rotated : np.ndarray
        A 3xN array (shape (3, N)) of wind vectors rotated into the LEC reference frame,
        with the y-axis aligned to the geographic North.
    
    Raises
    ------
    ValueError
        If 'wind' does not have shape (3, N).
    ValueError
        If the azimuth is outside the range [0, 360].
    ValueError
        If the anemometer model is not recognized (i.e., not "RM_YOUNG_81000" or "CAMPBELL_CSAT3").    
    
    Notes
    -----
    The function applies two sequential rotations:
    1. A model-dependent transformation that maintains the Cartesian reference frame while 
    ensuring that the u and v wind components are positive when aligned with the x- and y-axes, respectively.
    2. A rotation aligning the y-axis to the North, according to the specified azimuth.
    """
    
    # ROTATION from the intrinsic reference system of the instrument
    # to a standard reference frame with:
    # - u positive if aligned with the x-axis
    # - v positive if aligned with the y-axis
    # The y-axis is oriented in the direction defined by the azimuth angle relative to North (positive clockwise)
    # The rotation matrix depends on the anemometer model used.

    # --- Input validation ---
    # Ensure the 'wind' array has 3 rows (representing 3 components: u, v, w)
    if wind.shape[0] != 3:
        raise ValueError("'wind' must have shape (3, N)")

    # Ensure the azimuth value is within the valid range [0, 360]
    if not (0 <= azimuth <= 360):
        raise ValueError("azimuth is outside the range [0,360].")

    # Initialize a 3x3 rotation matrix depending on the anemometer model
    rot_model = np.zeros((3, 3))

    # Set rotation matrix based on the model
    if model == "RM_YOUNG_81000":
        rot_model[0, 0] = -1
        rot_model[1, 1] = -1
        rot_model[2, 2] = 1
    elif model == "CAMPBELL_CSAT3":
        rot_model[0, 1] = 1
        rot_model[1, 0] = -1
        rot_model[2, 2] = 1
    else:
        raise ValueError(f"Unknown model: {model}")

    # --- Rotation to LEC system with y-axis oriented to North ---
    # Convert azimuth from degrees to radians
    azimuth = np.deg2rad(azimuth)

    # Initialize rotation matrix for azimuth
    rot_azimuth = np.zeros((3, 3))

    # Calculate the cosine and sine of the azimuth angle
    cos_azimuth = np.cos(azimuth)
    sin_azimuth = np.sin(azimuth)

    # Define the rotation matrix for the azimuth transformation
    rot_azimuth[0, 0] = cos_azimuth
    rot_azimuth[0, 1] = sin_azimuth
    rot_azimuth[1, 0] = -sin_azimuth
    rot_azimuth[1, 1] = cos_azimuth
    rot_azimuth[2, 2] = 1

    # Combine the azimuth rotation and the model-specific rotation matrix
    rot_total = np.matmul(rot_azimuth, rot_model)

    # Apply the total rotation matrix to the wind data
    wind_rotated = np.matmul(rot_total, wind)

    return wind_rotated


def rotation_to_streamline_reference(wind : np.ndarray,
                                     wind_averaged : np.ndarray
                                     ) -> np.ndarray:
    """
    Rotate wind velocity components into the streamline coordinate system,
    using the double rotation method described in Kaimal and Finnigan (1979).

    This technique aligns the coordinate system with the average wind direction,
    such that:
    - the streamwise component (ũ) approximates the total wind speed,
    - the crosswise (ṽ) and vertical (w̃) components are minimized.

    The rotation is defined at each instant using the average wind vector,
    removing the mean crosswind and vertical components and aligning the flow
    with the x-axis of the new reference frame.

    Parameters
    ----------
    wind : np.ndarray
        Instantaneous wind velocity components of shape (3, N),
        where the first index represents (u, v, w).
    wind_averaged : np.ndarray
        Averaged (mean) wind velocity components of shape (3, N),
        used to define the streamline reference frame at each instant.

    Returns
    -------
    wind_rotated : np.ndarray
        Wind velocity components rotated into the streamline coordinate system,
        of shape (3, N).

    Raises
    ------
    ValueError
        If 'wind' or 'wind_averaged' do not have shape (3, N).
    ValueError
        If 'wind' and 'wind_averaged' do not have the same number of columns (N).

    Notes
    -----
    This method is most appropriate for stationary signals, where the mean wind
    vector is well defined and stable over time.

    See Also
    --------
    For theoretical background and mathematical details, see the documentation page:
    "Frame rotation and wind direction".

    References
    ----------
    Kaimal, J. C., & Finnigan, J. J. (1979). 
    Atmospheric Boundary Layer Flows. 
    Oxford University Press.
    """
    # --- Input validation ---
    if wind.shape[0] != 3 or wind_averaged.shape[0] != 3:
        raise ValueError("Both 'wind' and 'wind_averaged' must have shape (3, N)")
    if wind.shape[1] != wind_averaged.shape[1]:
        raise ValueError("'wind' and 'wind_averaged' must have the same number of columns (N)")

    N = wind.shape[1]
    u_averaged = wind_averaged[0, :]
    v_averaged = wind_averaged[1, :]
    w_averaged = wind_averaged[2, :]
    
    s = np.sqrt(u_averaged**2 + v_averaged**2)  # horizontal speed
    theta = np.arctan2(v_averaged, u_averaged)  # azimuth angle
    phi = np.arctan2(w_averaged, s)             # elevation angle

    # Build rotation matrices
    rot = np.zeros((3, 3, N))
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rot[0, 0, :] = cos_phi * cos_theta
    rot[0, 1, :] = cos_phi * sin_theta
    rot[0, 2, :] = sin_phi
    rot[1, 0, :] =-sin_theta
    rot[1, 1, :] = cos_theta
    rot[1, 2, :] = 0.0
    rot[2, 0, :] =-sin_phi * cos_theta
    rot[2, 1, :] =-sin_phi * sin_theta
    rot[2, 2, :] = cos_phi

    # Apply rotation
    wind_rotated = np.einsum('ijk,jk->ik', rot, wind)

    return wind_rotated

def wind_dir_LEC_reference(u : Union[np.ndarray, list, float, int], 
                           v : Union[np.ndarray, list, float, int],
                           threshold : float = 0.0
                           ) -> np.ndarray:
    """
    Compute wind direction from `u` and `v` wind components defined in a Local Earth Coordinate (LEC) reference system, 
    following the meteorological convention.

    Wind direction is defined as:
    - 0 degrees: wind coming from North
    - 90 degrees: wind coming from East
    - 180 degrees: wind coming from South
    - 270 degrees: wind coming from West

    If the wind magnitude is below a specified threshold, the function returns `NaN`.


    Parameters
    ----------
    u : array_like or scalar
        East-West wind component (positive towards East) in the LEC reference system.
    v : array_like or scalar
        North-South wind component (positive towards North) in the LEC reference system.
    threshold : float, optional
        Minimum horizontal wind speed modulus below which the wind direction is set to NaN. Default is 0.0.

    Returns
    -------
    wind_direction : np.ndarray
        Wind direction in degrees, values between 0° and 360° (0 inclusive, 360 exclusive), 
        or NaN if wind speed modulus is below the given threshold.

    Raises
    ------
    ValueError
        If `u` and `v` are arrays and their shapes do not match.
    ValueError
        If `threshold` is negative.
    """
    u = np.asarray(u)
    v = np.asarray(v)
    # --- Input validation ---
    if u.shape != v.shape:
        raise ValueError(f"Shape mismatch: u.shape = {u.shape}, v.shape = {v.shape}")
    if threshold < 0:
        raise ValueError(f" Threshold must be positive.")
    # Calculate the wind speed as the Euclidean norm of the u and v components
    wind_direction = (np.degrees(np.arctan2(u, v)) + 180) % 360
    # Apply thresholding to wind direction
    # If the wind speed is below the specified threshold, the wind direction is set to NaN.
    # This step ignores directions where the wind speed is too low to be meaningful.
    wind_speed = np.sqrt(u**2 + v**2)
    wind_direction = np.where(wind_speed < threshold, np.nan, wind_direction)

    return wind_direction


def wind_dir_modeldependent_reference(u : Union[np.ndarray, list, float, int],
                                      v : Union[np.ndarray, list, float, int],
                                      azimuth : float,
                                      model : str,
                                      threshold : float = 0.0
                                      ) -> np.ndarray:
    """
    Compute the wind direction based on the horizontal wind components (`u` and `v`), 
    defined in the proprietary coordinate system of the anemometer specified by the `model` argument. 
    The anemometer's head is oriented with a given `azimuth` angle with respect to the North.

    The function calculates the wind direction following the meteorological convention.

    Wind direction is defined as:
    - 0 degrees: wind coming from North
    - 90 degrees: wind coming from East
    - 180 degrees: wind coming from South
    - 270 degrees: wind coming from West . 
    
    If the wind magnitude is below a specified threshold, the function returns `NaN`.

    Parameters
    ----------
    u : array-like
        The u-component of the wind (east-west direction) in the anemometer's coordinate system.
    v : array-like
        The v-component of the wind (north-south direction) in the anemometer's coordinate system.
    azimuth : float
        The azimuth rotation in degrees to adjust the wind direction (e.g., instrument mounting offset).
    model : str
        The model of the anemometer, either "RM_YOUNG_81000" or "CAMPBELL_CSAT3".
    threshold : float, optional
        Minimum horizontal wind speed modulus below which the wind direction is set to NaN. Default is 0.0.

    Returns
    -------
    wind_direction : np.ndarray
        The wind direction in degrees, with 0° corresponding to North, 90° to East, etc., 
        or NaN if wind speed modulus is below the given threshold.

    Raises
    ------
    ValueError
        If the shapes of u and v do not match.
    ValueError
        If an unknown model is specified.
    ValueError
        If `threshold` is negative.
    """
    u = np.asarray(u)
    v = np.asarray(v)
    # --- Input validation ---
    if u.shape != v.shape:
        raise ValueError(f"Shape mismatch: u.shape = {u.shape}, v.shape = {v.shape}")
    if threshold < 0:
        raise ValueError(f" Threshold must be positive.")

    # Convert the model name to uppercase to ensure case-insensitivity
    model = model.upper()

    # --- Set the wind components based on the anemometer model ---
    # Based on the model, adjust the wind components (u, v) to align with the LEC system.
    if model == "RM_YOUNG_81000":
        u_LEC = -u   # Reverse the direction of the u component
        v_LEC = -v   # Reverse the direction of the v component
    elif model == "CAMPBELL_CSAT3":
        u_LEC = v    # Swap the u and v components
        v_LEC = -u   # Reverse the direction of the u component
    else:
        # Raise an error if the model is not recognized
        raise ValueError(f"Unknown model: {model}. Supported models are 'RM_YOUNG_81000' and 'CAMPBELL_CSAT3'.")

    # --- Calculate the wind direction in the LEC system ---
    # Calculate the wind direction using the atan2 function, consider the azimuth, 
    # use the meteorological convention, then convert it to degrees.
    wind_direction = (np.degrees(np.arctan2(u_LEC, v_LEC)) + azimuth + 180) % 360

    # --- Calculate the wind speed ---
    # The wind speed is computed as the Euclidean norm of the u and v components.
    wind_speed = np.sqrt(u**2 + v**2)

    # --- Thresholding ---
    # If the wind speed is below the specified threshold, set the true wind direction to NaN.
    # This step ignores directions where the wind speed is too low to be meaningful.

    wind_direction = np.where(wind_speed < threshold, np.nan, wind_direction)

    # return true_wind_direction
    return wind_direction

