import numpy as np
import pandas as pd

# Output directory and file name
dir_out = ""
name_out = "test_data"

# Overall parameters
length = 60  # Total duration [minutes]
sampl_frequency = 20  # Sampling frequency [Hz]
dt = int((1 / sampl_frequency) * 1000)  # Sampling interval [ms]
n_rows = length * 60 * sampl_frequency + 1  # Total number of samples

# Generate normally distributed synthetic data
# Wind components u, v (mean 2 m/s), w (mean 0.01 m/s), sonic temperature T_s (mean 20°C)
u = np.random.normal(loc=2, scale=1, size=n_rows)
v = np.random.normal(loc=2, scale=1, size=n_rows)
w = np.random.normal(loc=0.01, scale=1, size=n_rows)
T_s = np.random.normal(loc=20, scale=1, size=n_rows)

# Create time index for the data
time_start = pd.to_datetime("2012-09-28 02:00:00")
time_end = time_start + pd.Timedelta(minutes=length)
time_array = pd.date_range(start=time_start, end=time_end, freq=f"{dt}ms")

# Combine into a DataFrame
df = pd.DataFrame({"Time": time_array, "u": u, "v": v, "w": w, "T_s": T_s})
df["Time"] = pd.to_datetime(df["Time"])
df = df.set_index("Time")

# -----------------------------------------------
# Introduce specific anomalies into the dataset
# -----------------------------------------------

# 1. Remove timestamps to simulate missing data (10 seconds)
to_remove = pd.date_range(start="2012-09-28 02:05:00.000",
                          end="2012-09-28 02:05:10.000",
                          freq=f"{dt}ms")
df = df.drop(to_remove)

# 2. Add unrealistic high values (outliers) in 'u'
over_01 = pd.date_range(start="2012-09-28 02:30:00.000",
                        end="2012-09-28 02:30:00.100",
                        freq=f"{dt}ms")
df.loc[over_01, "u"] = 100

# 3. Add spike anomalies in 'u'

# Single positive spike
spike_01 = pd.date_range(start="2012-09-28 02:15:00.000",
                         end="2012-09-28 02:15:00.000",
                         freq=f"{dt}ms")
df.loc[spike_01, "u"] = 20

# Single negative spike
spike_01_b = pd.date_range(start="2012-09-28 02:16:00.000",
                           end="2012-09-28 02:16:00.000",
                           freq=f"{dt}ms")
df.loc[spike_01_b, "u"] = -20

# Three consecutive spikes
spike_02 = pd.date_range(start="2012-09-28 02:25:00.000",
                         end="2012-09-28 02:25:00.100",
                         freq=f"{dt}ms")
df.loc[spike_02, "u"] = 20

# Four consecutive spikes
spike_03 = pd.date_range(start="2012-09-28 02:35:00.000",
                         end="2012-09-28 02:35:00.150",
                         freq=f"{dt}ms")
df.loc[spike_03, "u"] = 20

# 4. Insert missing values (NaN) in 'u' over 5 seconds
nan_01 = pd.date_range(start="2012-09-28 02:40:00.000",
                       end="2012-09-28 02:40:05.000",
                       freq=f"{dt}ms")
df.loc[nan_01, "u"] = np.nan

# Export the DataFrame to a CSV file
df.index.name = "Time"
df.to_csv(f"{dir_out}{name_out}.csv",
          na_rep="NaN",
          float_format="%.7e")
