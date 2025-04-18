# External Modules
import numpy as np
import pandas as pd
import matplotlib as plt
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
logger.info("ProMeteo")

# parser definition to import the parameters of the run
params = core.load_config(config_file_path)
rawdata_path = params['rawdata_path']
dir_out = params['dir_out']
sampling_freq = params['sampling_freq']
horizontal_threshold=params['horizontal_threshold']
vertical_threshold=params['vertical_threshold']
temperature_threshold=params['temperature_threshold']

logger.info(f"""
            Parameters read:
            - Sampling Frequency: {sampling_freq}
            - Horizontal threshold: {horizontal_threshold}
            - Vetical threshold: {vertical_threshold}
            - Temperature threshold: {temperature_threshold}
            """)

# data import: it has to be a .csv file containing 4 columns: TIMESTAMP, u,v,w,T_s => data
rawdata=core.import_data(rawdata_path)
logger.info(f"""
            Raw Data imported from: {rawdata_path}
            """)

# filling holes known sampling frequency
data=pre_processing.fill_missing_timestamps(rawdata, sampling_freq)
logger.info(f"Missing timestamps filling completed.")

# non physical value cutting
data=pre_processing.remove_beyond_threshold(data,
                                            horizontal_threshold,
                                            vertical_threshold,
                                            temperature_threshold)
logger.info(f"Removing non physical values (over threshold) completed.")

# despiking

# salvataggio intermedio

# computation of wind direction

# salvataggio intermedio con colonna wind_dir

# rotation to streamline coordinate system

# salvataggio intermedio con colonna wind_dir

# reynolds decomposition => new dataframe containing mean(u), mean(v), mean(w), mean(T_s), mean(wind_dir), u', v', w', T_s', u'u', v'v', w'w', TKE, u'w', w'T'

# variables plotting