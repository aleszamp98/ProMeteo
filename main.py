import src.core as core
import src.pre_processing as pre_processing
import src.reynolds as reynolds
import src.plotting as plotting

path="/home/zampa/projects/ProMeteo/data/test_data.csv"

# parser definition to import hte parameters of the run

# creating the log file through a function

# data import: it has to be a .csv file containing 4 columns: TIMESTAMP, u,v,w,T_s => data
raw_data=core.import_data(path)

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