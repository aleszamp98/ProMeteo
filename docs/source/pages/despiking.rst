Despiking
=========

Despiking: identifies and corrects spikes in the data.

Two methods:

VM97 (Vickers & Mahrt, 1997):

Iterative.

Spikes identified using mean and standard deviation.

Short isolated anomalies are replaced via interpolation.

Robust (custom):

Non-iterative.

Spikes identified via median and interquartile range.

Faster on long time series.

NaNs (from threshold removal or missing data) are linearly interpolated at the end.
