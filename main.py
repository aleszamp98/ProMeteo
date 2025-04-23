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
import src.reynolds as reynolds
import src.plotting as plotting


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
horizontal_threshold = params['horizontal_threshold']
vertical_threshold = params['vertical_threshold']
temperature_threshold = params['temperature_threshold']
despiking_mode = params['despiking_mode']
window_length_despiking = params['window_length_despiking']
max_length_spike = params['max_length_spike']
max_iterations = params['max_iterations']
c_H = params['c_H']
c_V = params['c_V']
c_T = params['c_T']


# data import: it has to be a .csv file containing 4 columns: TIMESTAMP, u,v,w,T_s => data
rawdata=core.import_data(rawdata_path)
logger.info(f"""
            Selected sampling frequency: {sampling_freq}
            Raw Data imported from: {rawdata_path}
            """)

# filling missing timestamps known sampling frequency
data=pre_processing.fill_missing_timestamps(rawdata, sampling_freq)
logger.info(f"""
            Missing timestamps filling completed.
            """)

# non physical value cutting
logger.info(f"""
            Removing values from time series that exceed the following thresholds:
            - Horizontal threshold: {horizontal_threshold}
            - Vetical threshold: {vertical_threshold}
            - Temperature threshold: {temperature_threshold}
            """)
data=pre_processing.remove_beyond_threshold(data,
                                            horizontal_threshold,
                                            vertical_threshold,
                                            temperature_threshold)
logger.info(f"""
            Removing exceeding values completed.
            """)

# despiking
logger.info(f"""
            Running despiking.
            """)

window_length_despiking_points = core.min_to_points(sampling_freq, 
                                          window_length_despiking)
if window_length_despiking_points % 2 == 0:
    window_length_despiking_points += 1
data_despiked = pd.DataFrame(index=data.index, columns=data.columns)

if despiking_mode == "VM97":
    logger.info(f"""
                - Mode: {despiking_mode}
                - Moving window length: {window_length_despiking} min => {window_length_despiking_points} points
                - Maximum number of consecutive values to be considered spike: {max_length_spike}
                - Maximum number of iterations to perform: {max_iterations}
                - Starting values of c:
                    - For the horizontal components of the wind: {c_H}
                    - For the vertical component: {c_V}
                    - For the sonic temperature: {c_T}
                """)
    c_list = [c_H, c_H, c_V, c_T] # starting constants
    for col, c in zip(['u', 'v', 'w', 'T_s'], c_list):
        logger.info(f"""
                    Despiking variable {col}
                    """)
        array_to_despike = data[col].to_numpy()

        data_despiked[col] = pre_processing.despiking_VM97(array_to_despike,
                                                           c,
                                                           window_length_despiking_points,
                                                           max_length_spike,
                                                           max_iterations,
                                                           logger
                                                           )
elif despiking_mode == "robust":
    logger.info(f"""
                - Mode: {despiking_mode}
                - Moving window length: {window_length_despiking} min => {window_length_despiking_points} points
                """)
    for col in ['u', 'v', 'w', 'T_s']:
        array_to_despike = data[col].to_numpy()
        data_despiked[col] = pre_processing.despiking_ROBUST(array_to_despike,
                                                             window_length_despiking_points)

# comparison between non-despiked and despiked time series



del data # cleaning environment



# salvataggio intermedio

# computation of wind direction

# salvataggio intermedio con colonna wind_dir

# rotation to streamline coordinate system

# salvataggio intermedio con colonna wind_dir

# reynolds decomposition => new dataframe containing mean(u), mean(v), mean(w), mean(T_s), mean(wind_dir), u', v', w', T_s', u'u', v'v', w'w', TKE, u'w', w'T'

# variables plotting