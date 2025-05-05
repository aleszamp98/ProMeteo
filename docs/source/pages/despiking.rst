Despiking
=========

Data spikes can be caused by random electronic noise in the monitoring or recording systems, 
as might occur during precipitation when water collects on the transducers of sonic anemometers. 
These spikes are unwanted outliers that can strongly affect the computation of averages 
and derived fluxes, and should therefore be removed. Unlike clearly non-physical outliers, 
which can often be identified using fixed thresholds (e.g., wind speeds exceeding 50 m/s 
when values no greater than 10 m/s are expected under certain conditions), 
spikes cannot be reliably detected with a single predefined threshold. 
For this reason, it is necessary to adopt moving-window-based methods, 
where thresholds are dynamically computed based on the local mean or median 
and on an acceptable range determined by the variability of the time series 
around a specific time point.

The despiking procedure is useful to filter unwanted spikes, 
i.e. a single or a small group of data points characterised 
by a magnitude too different compared to the data distribution. 
The despiking ensures that the averaging procedure will not be 
affected by these spikes. The presence of spikes in the measured 
variables can also increase the error of the derived quantities.

The ProMeteo library implements two despiking procedures to identify and replace these spikes: 
one based on the method described by Vickers and Mahrt [Vickers1997]_, and another, simpler, 
based on robust statistical metrics. 
These are available as functions in the ``pre_processing`` module:

- ``pre_processing.despiking_VM97()`` for the Vickers and Mahrt (1997) method,
- ``pre_processing.despiking_robust()`` for the custom "robust" method.

The script ``main.py`` executes one of the two available methods based 
on the ``despiking_method`` parameter ("VM97" or "robust") specified in the configuration file ``config/config.txt``.


Vickers and Mahrt (1997) Method
-------------------------------

The despiking procedure is similar to the one described in Vickers and Mahrt [Vickers1997]_. 
This method assumes that each interval within the dataset follows a Gaussian distribution of independent data 
characterised by mean (:math:`\mu`) and standard deviation (:math:`\sigma`).
The method uses a centred moving window of size :math:`N`, defined by the parameter ``window_length``, 
to compute two thresholds called ``upper_bound`` and ``lower_bound``.
The window is moved along the time series, and at each step, the mean and standard deviation are computed.

.. math::

    \mu = \frac{1}{N} \sum_i^N \zeta_i, \quad
    \sigma = \sqrt{ \frac{1}{N} \sum_i^N (\zeta_i - \mu)^2 } ,

where :math:`\zeta_i` is the value of the time series at time :math:`i`.

The upper and lower bounds are defined as:

.. math::

    upper(lower)_{bound} = \mu \pm c \cdot \sigma,

where :math:`c` is a user-defined parameter that determines the width around 
the mean within which values are considered acceptable.

Values exceeding :math:`upper(lower)_{bound}` are marked as possible spikes.
Only sequences of consecutive values exceeding the thresholds with a length up to the parameter ``max_consecutive_spikes`` are 
cosidered spikes and therofore replaced.
Each spike is replaced by linear interpolation from its closest neighbors, when possible.
The process is repeated iteratively, increasing :math:`c` by 0.1 at each iteration, 
until no further spikes are detected or the maximum number of iterations is reached.
This increasing of :math:`c` allows the method to converge to the condition of no spikes detected.
The maximum number of iterations is defined by the parameter ``max_iterations``.

The method is applied by the ``main.py`` script separately to each variable, 
using the corresponding ``c`` factors specified in the configuration file:

- for horizontal wind components (:math:`u`, :math:`v`): ``c_H``,
- for vertical wind component (:math:`w`): ``c_V``,
- for sonic temperature (:math:`T_s`): ``c_T``.

It is recommended to use:

- A window of 5 minutes for 20 Hz data or 30 minutes for 1 Hz data
- :math:`c = 3.5` for horizontal wind components (:math:`u`, :math:`v`) and sonic temperature (:math:`T_s`)
- :math:`c = 5` for vertical wind component (:math:`w`) due to its smaller magnitude and higher variability.

Workflow of ``pre_processing.despiking_VM97()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The function ``pre_processing.despiking_VM97()`` uses ``core.running_stats()`` to compute 
the moving average and variance over a sliding window. It then calculates the corresponding 
upper and lower bounds and uses them to create a mask of candidate spikes (the ones exceeding these thresholds).

This mask is passed to the function ``pre_processing.identify_interp_spikes()``, which identifies sequences of consecutive 
spikes that do not exceed the maximum allowed length, discarding those that are too long.

For the remaining valid spikes, interpolation is performed using ``pre_processing.linear_interp()``, 
but only when the nearest neighbors are non-NaN values and the spike is not located near the edges of the time series.

"Robust" Method
---------------

This method is based on robust statistics and does not assume Gaussianity. 
It is designed to be non-iterative and thus computationally faster.

The method uses a centred moving window of size :math:`N`, defined by the parameter ``window_length``, 
to compute two thresholds called ``upper_bound`` and ``lower_bound``.
The window is moved along the time series, and at each step,
the median :math:`\tilde{\mu}` and the interquartile range :math:`\tilde{\sigma}` are computed:

.. math::

    \tilde{\mu} = \text{median} \quad
    \tilde{\sigma} = \frac{q_{84} - q_{16}}{2},

where :math:`q_{84}` and :math:`q_{16}` are the 84th and 16th percentiles.

The upper and lower bounds are defined as:

.. math::

    upper(lower)_{bound} = \max\left\{ \tilde{\mu} \pm c \cdot \tilde{\sigma},\ 0.5 \right\}.

Outliers (isolated points or sequences greater than :math:`upper_{bound}` or lower than :math:`lower_{bound})` 
are considered spikes. They are replaced by the median value computed over the same window.

The :math:`c` factor is defined in the configuration file as ``c_robust``.
The same recommendations as for the Vickers and Mahrt method apply.

Workflow of ``pre_processing.despiking_robust()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The function ``pre_processing.robust()`` calls ``core.running_stats_robust()`` to compute robust statistics 
over a sliding window. It then calculates the corresponding ``upper_bound`` and ``lower_bound`` values.

A mask is created to identify all data points falling outside these bounds. 
These values are then replaced with the corresponding entries from the time series of moving medians.

Comparison between the Two Methods
----------------------------------

+------------------------+--------------------------------------------+--------------------------------------------+
| **Criterion**          | **despiking_VM97()**                       | **despiking_robust()**                     |
+========================+============================================+============================================+
| Definition of spike    | A spike is a point or short sequence       | A spike is any point or sequence outside   |
|                        | (up to `max_consecutive_spikes`) outside   | the acceptable range, with no constraint   |
|                        | the acceptable range.                      | on sequence length.                        |
+------------------------+--------------------------------------------+--------------------------------------------+
| Acceptable range       | Defined by mean and standard deviation     | Defined by median and interquartile range. |
|                        | over a moving window.                      |                                            |
+------------------------+--------------------------------------------+--------------------------------------------+
| Spike replacement      | Replaced with linearly interpolated values | Replaced with the moving median at the     |
|                        | from neighbors.                            | corresponding time.                        |
+------------------------+--------------------------------------------+--------------------------------------------+
| Type of process        | Iterative.                                 | Non-iterative.                             |
+------------------------+--------------------------------------------+--------------------------------------------+
| Expected behavior      | More conservative: detects fewer spikes,   | Less conservative: flags any outlier point |
|                        | limited to short sequences. May require    | or sequence. Faster due to non-iterative   |
|                        | more computation time on long time series. | structure and no interpolation.            |
+------------------------+--------------------------------------------+--------------------------------------------+
| Literature presence    | Described in scientific literature.        | Proprietary method.                        |
+------------------------+--------------------------------------------+--------------------------------------------+

References
----------

.. [Vickers1997] Vickers, D., & Mahrt, L. (1997). *Quality Control and Flux Sampling Problems for Tower and Aircraft Data*. Journal of Atmospheric and Oceanic Technology, 14(3), 512–526. https://doi.org/10.1175/1520-0426(1997)014<0512:QCAFSP>2.0.CO;2

