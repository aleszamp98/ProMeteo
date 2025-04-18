# External Modules
import numpy as np
import pandas as pd
import matplotlib as plt
import os

# ProMeteo Modules
import src.core as core
import src.pre_processing as pre_processing
import src.reynolds as reynolds
import src.plotting as plotting

# configuration-file path (meant to be in the "config/" folder)
script_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.abspath(os.path.join(script_dir, 'config', 'config.txt'))

# parser definition to import hte parameters of the run
params = core.load_config(config_file_path)
rawdata_path = params['rawdata_path']
dir_out = params['dir_out']
sampling_freq = params['sampling_freq']

# creating the log file through a function

# data import: it has to be a .csv file containing 4 columns: TIMESTAMP, u,v,w,T_s => data
raw_data=core.import_data(rawdata_path)

# filling holes known sampling frequency

# non physical value cutting

# despiking

# salvataggio intermedio

# computation of wind direction

# salvataggio intermedio con colonna wind_dir

# rotation to streamline coordinate system

# salvataggio intermedio con colonna wind_dir

# reynolds decomposition => new dataframe containing mean(u), mean(v), mean(w), mean(T_s), mean(wind_dir), u', v', w', T_s', u'u', v'v', w'w', TKE, u'w', w'T'

# variables plotting