# External Modules
import numpy as np
import pandas as pd
import os
import logging

# ProMeteo Modules
import src.core as core
import src.pre_processing as pre_processing
import src.frame as frame

def main():
    # === Logger Setup ===
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('data/run.log', mode='w')
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Initializing logger
    logger.info("""
            ProMeteo
            """)

    # === Load Configuration Parameters ===
    # Load parameters from the configuration file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(script_dir, 'config/config.txt')
    params = core.load_config(config_file_path)
    # Make input and output paths relative to script location
    rawdata_path = os.path.join(script_dir, params['rawdata_path'])
    dir_out = os.path.join(script_dir, params['dir_out'])
    
    sampling_freq = params['sampling_freq']
    model = params['model']
    horizontal_threshold = params['horizontal_threshold']
    vertical_threshold = params['vertical_threshold']
    temperature_threshold = params['temperature_threshold']
    despiking_method = params['despiking_method']
    window_length_despiking = params['window_length_despiking']
    max_length_spike = params['max_length_spike']
    max_iterations = params['max_iterations']
    c_H = params['c_H']
    c_V = params['c_V']
    c_T = params['c_T']
    c_robust = params['c_robust']
    window_length_averaging = params['window_length_averaging']
    reference_frame = params['reference_frame']
    azimuth = params['azimuth']

    # === Data Import ===
    # Import raw data from the specified path, expecting a CSV file with 4 columns: Time, u,v,w,T_s
    rawdata = core.import_data(rawdata_path)
    logger.info(f"""
            Raw Data imported from: {rawdata_path}
            Anemometer model: {model}
            Selected sampling frequency: {sampling_freq}
            """)
    col_list = ['u', 'v', 'w', 'T_s']

    # === Fill Missing Timestamps ===
    # Fill missing timestamps based on the known sampling frequency
    data = pre_processing.fill_missing_timestamps(rawdata, sampling_freq)
    logger.info(f"""
            Missing timestamps filling completed:
            Number of timestamps in the rawdata package: {rawdata.shape[0]}
            Number of timestamps after the filling procedure: {data.shape[0]}
            """)
    del rawdata

    # === Remove Non-physical Values ===
    # Remove any data points beyond the threshold values for each time series
    threshold_list = [horizontal_threshold, horizontal_threshold, vertical_threshold, temperature_threshold]
    data_cleaned = pd.DataFrame(index=data.index, columns=data.columns)
    
    for col, threshold in zip(col_list, threshold_list):
        array = data[col].to_numpy()
        # Remove values that are beyond the threshold for each component
        data_cleaned[col], count = pre_processing.remove_beyond_threshold(array, threshold)
        logger.info(f"""
                {count} points beyond {threshold} replaced with NaNs from '{col}' time series
                """)
    del data  # cleaning environment

    # === Despiking ===
    # Apply despiking method to remove spikes from the data
    logger.info(f"""
            Despiking
            """)
    window_points = core.min_to_points(sampling_freq, window_length_despiking)
    data_despiked = pd.DataFrame(index=data_cleaned.index, columns=data_cleaned.columns)

    if despiking_method == "VM97":
        # VM97 Despiking Method
        logger.info(f"""
                - Mode: {despiking_method}
                - Moving window length: {window_length_despiking} min => {window_points} points
                - Maximum number of consecutive values to be considered spike: {max_length_spike}
                - Maximum number of iterations to perform: {max_iterations}
                - Starting values of c:
                    - For the horizontal components of the wind: {c_H}
                    - For the vertical component: {c_V}
                    - For the sonic temperature: {c_T}
                """)
        for col, c in zip(col_list, [c_H, c_H, c_V, c_T]):
            array = data_cleaned[col].to_numpy()
            data_despiked[col] = pre_processing.despiking_VM97(array, c, window_points, max_length_spike, max_iterations, logger)
            replaced = np.sum(array != data_despiked[col].to_numpy())
            logger.info(f"""
                    Number of modified values: {replaced} 
                    """)
    elif despiking_method == "robust":
        # Robust Despiking Method
        logger.info(f"""
                - Mode: {despiking_method}
                - Moving window length: {window_length_despiking} min => {window_points} points
                - Selected constant: {c_robust}
                """)
        for col in col_list:
            array = data_cleaned[col].to_numpy()
            data_despiked[col], count = pre_processing.despiking_robust(array, c_robust, window_points)
            logger.info(f"""
                    Number of modified values: {count} 
                    """)
    del data_cleaned  # cleaning environment

    # === Interpolate Missing Values ===
    # Interpolate NaNs that were created during the despiking process or already present in rawdata
    data_interp = pd.DataFrame(index=data_despiked.index, columns=data_despiked.columns)
    for col in col_list:
        array = data_despiked[col].to_numpy()
        data_interp[col], interpolated = pre_processing.interp_nan(array)
        remaining_nans = np.isnan(data_interp[col]).sum()
        logger.info(f"""
                NaNs interpolation of '{col}' time series:
                - Number of interpolated NaNs: {interpolated}
                - Number of remaining NaNs: {remaining_nans}
                """)
    del data_despiked  # cleaning environment

    # === Save Preprocessed Data ===
    # Save the preprocessed data to a CSV file
    data_interp.index.name = "Time"
    data_interp.to_csv(os.path.join(dir_out, "data_preprocessed.csv"),
                       na_rep='NaN', float_format='%.7e', index=True)
    logger.info(f"""
            Pre-processed data saved.
            """)

    # === Reference Frame Rotation & Wind Direction Computation ===
    # Rotate the data to the chosen reference frame (e.g., LEC or streamline) and calculate wind direction
    window_avg_points = core.min_to_points(sampling_freq, window_length_averaging)
    logger.info(f"""
            Rotation to {reference_frame} reference frame
            and Wind Direction computation.
            - Moving window length: {window_length_averaging} min => {window_avg_points} points
            """)

    wind = np.array([data_interp['u'].to_numpy(),
                     data_interp['v'].to_numpy(),
                     data_interp['w'].to_numpy()])
    wind_averaged = np.zeros_like(wind)

    if reference_frame == "LEC":
        # Rotation to Local Earth Coordinate (LEC) system
        wind_rotated = frame.rotation_to_LEC_reference(wind, azimuth, model)
        for i in range(3):
            wind_averaged[i, :], _ = core.running_stats(wind_rotated[i, :], window_avg_points)
        wind_direction = frame.wind_dir_LEC_reference(wind_averaged[0], wind_averaged[1])
    elif reference_frame == "streamline":
        # Rotation to streamline reference frame
        for i, comp in enumerate(['u', 'v', 'w']):
            wind_averaged[i, :], _ = core.running_stats(data_interp[comp].to_numpy(), window_avg_points)
        wind_rotated = frame.rotation_to_streamline_reference(wind, wind_averaged)
        wind_direction = frame.wind_dir_modeldependent_reference(wind_averaged[0], wind_averaged[1], azimuth, model)
        del wind_averaged  # Clean up

    # === Save Rotated Data ===
    # Save the rotated wind components and the calculated wind direction to a CSV file
    data_rotated = pd.DataFrame(index=data_interp.index)
    for i, comp in enumerate(['u', 'v', 'w']):
        data_rotated[comp] = wind_rotated[i]
    data_rotated["T_s"] = data_interp["T_s"]
    data_rotated["wind_dir"] = wind_direction
    data_rotated.index.name = "Time"
    data_rotated.to_csv(os.path.join(dir_out, f"data_preprocessed_rotated_{reference_frame}.csv"),
                        na_rep='NaN', float_format='%.7e', index=True)
    del data_interp, wind, wind_direction  # Clean up
    logger.info(f"""
            Rotated wind components saved.
            """)

if __name__ == "__main__":
    main()