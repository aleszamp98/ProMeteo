.. _test-data:

Test data (``test_data.csv``)
=============================

Within the ``data/`` directory, the script ``generate_data.py`` creates a synthetic ``.csv`` file 
that can be used as input for a run of ``main.py`` in the *ProMeteo* library. 
This dataset is designed to simulate a set of raw data sampled from an anemometer 
of unspecified model. The wind components are assumed to be defined in the proprietary 
sonic coordinate system of the instrument.

Only the ``u`` component of the wind is intentionally altered by introducing anomalies, 
with the goal of testing the capabilities of the ``pre_processing`` module. 
An example of how to use these features and the resulting effects on the raw data 
is provided in the section: ``usage_example``.


Variables
---------
All variables are generated from a normal distribution:

- ``u``, ``v``: mean = 2, std = 1
- ``w``: mean = 0.01, std = 1
- ``T_s``: mean = 20, std = 1

Timestamps
----------

- Time range: **2012-09-28 02:00:00** to **2012-09-28 03:00:00**
- Frequency: **20 Hz** (every 50 milliseconds)
- Total rows: **72,001**

Data Anomalies Introduced
-------------------------

**1. Missing Timestamps**

- Gap introduced from ``02:05:00.000`` to ``02:05:10.000``
- Duration: **10 seconds**
- Missing samples: **201**

**2. Extreme Values**

- ``u`` set to **100 m/s** (unrealistically high) from ``02:30:00.000`` to ``02:30:00.100``
- Duration: **100 milliseconds**
- Affected samples: **3**

**3. Spikes in ``u``**

- Single spike at ``02:15:00.000``: **+20 m/s**
- Single spike at ``02:16:00.000``: **–20 m/s**
- Three consecutive spikes from ``02:25:00.000`` to ``02:25:00.100``: **+20 m/s**
- Four consecutive spikes from ``02:35:00.000`` to ``02:35:00.150``: **+20 m/s**

**4. Missing Values**

- ``u`` set to **NaN** from ``02:40:00.000`` to ``02:40:05.000``
- Duration: **5 seconds**
- Missing samples: **101**

Output
------

- File: ``test_data.csv``
- Format: CSV with timestamp index
- Missing values are saved as ``"NaN"``