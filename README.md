# ProMeteo: PROcessing of METEOrological Data

<img src="img/logo.png" alt="Logo" width="200"/>

**ProMeteo** is a Python library for preprocessing and manipulating measurements mainly from sonic anemometers and other instruments mounted on meteorological towers.

It addresses specific needs of scientists and analysts working with data collected from meteorological towers. The functionalities of ProMeteo are modular — they can be used independently or together in a pipeline that goes from raw data to derived quantities and visualizations.

---

## Features
ProMeteo is capable of:

- Removing non-physical values from time series.
- Despiking time series using two methodologies:
  - *Vickers and Mahrt* (iterative)
  - A non-iterative method based on robust statistics
- Interpolating missing or removed values to obtain a continuous time series.
- Calculating wind direction from wind components in the intrinsic reference system of the sonic anemometer.
- Performing coordinate rotation into the streamline reference frame (as defined by Finnigan).
- Calculating first- (e.g. mean, fluctuations) and second-order statistics (e.g. variances, turbulent fluxes, TKE).
- Visualizing processed data in a simple and efficient way.

---

## Dependencies
To run ProMeteo, the following Python packages are required:

- `numpy`
- `pandas`
- `matplotlib`

You can install them with:

```bash
pip install numpy pandas matplotlib
```

# Documentation

The full documentation for the project is available on [Read the Docs](https://pro-meteo.readthedocs.io/en/latest/).


## Contributing
ProMeteo is an open project. Suggestions, corrections, and contributions are very welcome!

-Open an issue for bugs or feature requests
-Submit a pull request to contribute code or improvements
-Or contact me directly via email (see below)

## How to Cite
If you use ProMeteo in a publication or presentation, please cite it as:
```
@software{ProMeteo_zampella_alessandro,
  author       = {Alessandro Zampella},
  title        = {ProMeteo},
  month        = apr,
  year         = 2025,
  url          = {https://github.com/aleszamp98/ProMeteo.git}
}
```
