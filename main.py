# External Modules
import numpy as np
import pandas as pd
import matplotlib as plt
import sys
import os
import logging

# ProMeteo Modules
import src.core as core
import src.pre_processing as pre_processing
import src.frame as frame


# configuration-file path (meant to be in the "config/" folder)
script_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = f'{script_dir}/config/config.txt'

# setting the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(f'{script_dir}/data/run.log', mode='w')
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.info("""
            ProMeteo
            """)

# parser definition and import the parameters of the run
params = core.load_config(config_file_path)
rawdata_path = params['rawdata_path']
dir_out = params['dir_out']
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

# data import: it has to be a .csv file containing 4 columns: TIMESTAMP, u,v,w,T_s => rawdata
rawdata=core.import_data(rawdata_path)
logger.info(f"""
            Raw Data imported from: {rawdata_path}
            Anemometer model: {model}
            Selected sampling frequency: {sampling_freq}
            """)
col_list=['u', 'v', 'w', 'T_s'] #Time as index


#######################################################################
######################### pre-processing ##############################
#######################################################################

# filling missing timestamps known sampling frequency
data=pre_processing.fill_missing_timestamps(rawdata, sampling_freq)
logger.info(f"""
            Missing timestamps filling completed:
            Number of timestamps in the rawdata package: {rawdata.shape[0]}
            Number of timestamps after the filling procedure: {data.shape[0]}
            """)
del rawdata

# non physical value cutting
threshold_list = [horizontal_threshold,
                  horizontal_threshold,
                  vertical_threshold,
                  temperature_threshold]

data_cleaned = pd.DataFrame(index=data.index, columns=data.columns)

for col, threshold in zip(col_list, threshold_list):
    array_to_clean = data[col].to_numpy()
    data_cleaned[col], count_beyond = pre_processing.remove_beyond_threshold(array_to_clean,
                                                                             threshold)
    logger.info(f"""
            {count_beyond} points beyond {threshold} replaced with NaNs from '{col}' time series
            """)
    del array_to_clean, count_beyond
del data  # cleaning environment

# despiking
logger.info(f"""
            Despiking
            """)

window_length_despiking_points = core.min_to_points(sampling_freq, 
                                                    window_length_despiking)
if window_length_despiking_points % 2 == 0:
    window_length_despiking_points += 1

data_despiked = pd.DataFrame(index=data_cleaned.index, columns=data_cleaned.columns)

if despiking_method == "VM97":
    logger.info(f"""
            - Mode: {despiking_method}
            - Moving window length: {window_length_despiking} min => {window_length_despiking_points} points
            - Maximum number of consecutive values to be considered spike: {max_length_spike}
            - Maximum number of iterations to perform: {max_iterations}
            - Starting values of c:
                - For the horizontal components of the wind: {c_H}
                - For the vertical component: {c_V}
                - For the sonic temperature: {c_T}
                """)
    c_list = [c_H, c_H, c_V, c_T] # starting constants
    for col, c in zip(col_list, c_list):
        logger.info(f"""
            Despiking '{col}' time series
                    """)
        array_to_despike = data_cleaned[col].to_numpy()

        data_despiked[col] = pre_processing.despiking_VM97(array_to_despike,
                                                           c,
                                                           window_length_despiking_points,
                                                           max_length_spike,
                                                           max_iterations,
                                                           logger)
        # comparison between non-despiked and despiked time series
        replaced = np.sum(array_to_despike != data_despiked[col].to_numpy())
        logger.info(f"""
            Number of modified values: {replaced} 
            """)
        del array_to_despike, replaced

elif despiking_method == "robust":
    logger.info(f"""
            - Mode: {despiking_method}
            - Moving window length: {window_length_despiking} min => {window_length_despiking_points} points
            - Selected constant: {c_robust}
                """)
    for col in col_list:
        logger.info(f"""
            Despiking '{col}' time series
                    """)
        array_to_despike = data_cleaned[col].to_numpy()
        data_despiked[col], count_spike = pre_processing.despiking_robust(array_to_despike,
                                                                          c_robust,
                                                                          window_length_despiking_points)
        # comparison between non-despiked and despiked time series
        logger.info(f"""
            Number of modified values: {count_spike} 
            """) 
        del array_to_despike, count_spike
        
del data_cleaned # cleaning environment

# Nan interpolation

data_interp = pd.DataFrame(index=data_despiked.index, columns=data_despiked.columns)

for col in col_list:
    array_to_interp = data_despiked[col].to_numpy()
    data_interp[col], count_interp = pre_processing.interp_nan(array_to_interp)
    count_remaining_nans = np.sum(np.isnan(data_interp[col].to_numpy()))
    # comparison between before and after the interpolation procedure
    logger.info(f"""
        NaNs interpolation of '{col}' time series:
        - Number of interpolated NaNs: {count_interp}
        - Number of remaining NaNs: {count_remaining_nans}
        """) 
    del array_to_interp, count_interp, count_remaining_nans

del data_despiked

# saving preprocessed data
data_interp.index.name = "Time"
data_interp.to_csv(dir_out+"data_preprocessed.csv",
                   na_rep='NaN',
                   float_format='%.7e', 
                   index=True)

logger.info(f"""
            Pre-processed data saved.
            """)

#######################################################################
######## reference frame rotation & wind direction computation ########
#######################################################################

window_length_averaging_points = core.min_to_points(sampling_freq, 
                                                    window_length_averaging)
if window_length_averaging_points % 2 == 0:
    window_length_averaging_points += 1

logger.info(f"""
            Rotation to {reference_frame} reference frame
            and Wind Direction computation.
            - Moving window length: {window_length_averaging} min => {window_length_averaging_points} points
            """)

data_rotated = pd.DataFrame(index=data_interp.index, columns=data_interp.columns)
wind = np.array([data_interp['u'].to_numpy(),
                 data_interp['v'].to_numpy(),
                 data_interp['w'].to_numpy()])
wind_averaged = np.full((3, len(data_interp)), 0.0)

if reference_frame == "LEC":
    # ROTATION TO LEC (Local Earth Coordinate) System, given the type and azimuth of the instrument
    wind_rotated = frame.rotation_to_LEC_reference(wind,
                                                            azimuth,
                                                            model)
    for i, component in enumerate(['u','v','w']):
        wind_averaged[i,:],_ = core.running_stats(wind_rotated[i,:], # LEC wind components necessary in the wind_direction computation
                                                  window_length_averaging_points)
    wind_direction = frame.wind_dir_LEC_reference(wind_averaged[0,:],
                                                           wind_averaged[1,:]) # only horizontal components in LEC system needee
    
    # maintain in memory the wind_averaged array cause is necessary to the next step: reynolds decomposition
    
# rotazione in un riferimento streamline (indipendente da orientazione)
# riparte da componenti sistema intrinseco => compie media (può essere interna)
#  => esegue rotazione sulla base delle componenti medie
elif reference_frame == "streamline":
    for i, component in enumerate(['u','v','w']):
        wind_averaged[i,:],_ = core.running_stats(data_interp[component].to_numpy(), # non rotated wind components necessary in the wind_direction computation
                                                  window_length_averaging_points)
    wind_rotated = frame.rotation_to_streamline_reference(wind, # model indipendent
                                                                   wind_averaged)
    wind_direction = frame.wind_dir_modeldependent_reference(wind_averaged[0,:],
                                                                      wind_averaged[1,:],
                                                                      azimuth,
                                                                      model) # model dependent computation
    del wind_averaged # delete wind_averaged cause is in the old reference system, not useful to the next step: reynolds decomposition


# saving rotated data
for i, component in enumerate(['u','v','w']):
    data_rotated[component] = wind_rotated[i,:]
data_rotated["T_s"] = data_interp["T_s"] # copy the T_s column unchanged
data_rotated["wind_dir"] = wind_direction
data_rotated.index.name = "Time"
data_rotated.to_csv(dir_out+f"data_preprocessed_rotated_{reference_frame}.csv",
                   na_rep='NaN',
                   float_format='%.7e', 
                   index=True)
del data_interp
del wind, wind_direction

logger.info(f"""
            Rotated wind components saved.
            """)

#######################################################################
#######################################################################
#######################################################################