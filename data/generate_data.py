import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


dir_out=os.path.dirname(os.path.abspath(__file__))+"/"
name_out="test_data"
length=60 # [min]
sampl_frequency=20 # [Hz]
dt=int((1/sampl_frequency)*1000) # [ms]
n_rows= 60*60*20+1 # number of fictitious measures

# generates normal distributed values for the three components of the wind and the sonic temperature
u=np.random.normal(loc=2, scale=1, size=n_rows) # [m/s]
v=np.random.normal(loc=2, scale=1, size=n_rows) # [m/s]
w=np.random.normal(loc=0.01, scale=1, size=n_rows) # [m/s]
T_s=np.random.normal(loc=20, scale=1, size=n_rows) # [°C]
# generates an array of timestamps to be attached to the fictious measures
time_start=pd.to_datetime("2012-09-28 02:00:00")
time_end=time_start+pd.Timedelta(minutes=length)
time_array=pd.date_range(start=time_start, end=time_end, freq=f"{dt}ms")
df=pd.DataFrame({"Time":time_array, "u":u, "v":v, "w":w, "T_s":T_s})
df["Time"] = pd.to_datetime(df["Time"])
df = df.set_index("Time")

# values over threshold
first_over=pd.date_range(start="2012-09-28 02:30:00.000", end="2012-09-28 02:30:00.100", freq=f"{dt}ms")
df.loc[first_over, "u"]= 100

# spike
spike_01=pd.date_range(start="2012-09-28 02:15:00.000", end="2012-09-28 02:15:00.100", freq=f"{dt}ms")
df.loc[spike_01, "u"]= 10

df.index.name = "Time"
df.to_csv(f"{dir_out}{name_out}.csv",
            na_rep='NaN',
            float_format='%.7e')

plt.plot(df.index, df["u"])
plt.savefig("plot.png")