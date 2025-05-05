# ProMeteo: PROcessing of METEOrological Data

<img src="img/logo.png" alt="Logo" width="200"/>

**ProMeteo** is a Python library for preprocessing and manipulating measurements collected by sonic anemometers mounted on meteorological towers.

It addresses specific needs of scientists and analysts working with data collected from meteorological towers. The functionalities of ProMeteo are modular — they can be used independently or together in a pipeline that goes from raw data to processeed data and derived quantities.

---

## Features
ProMeteo is capable of:

- Filling missing timestamps in a time series, accordingly to a specified sampling frequency
- Removing non-physical values from time series.
- Despiking time series using two methodologies:
  - *Vickers and Mahrt, 1997* method
  - A costum method called "robust"
- Interpolating missing or removed values to obtain a continuous time series.
- Calculating wind direction from horizontal wind components.
- Performing wind components rotation, from the intrinsic sonic coordinate frame to the:
   - *streamline reference frame* (as defined by *Kaimal and Finnigan, 1994*).
   - *LEC (Local Earth Coordinate) reference frame*


# Documentation

The full documentation for the project is available on [Read the Docs: ProMeteo](https://pro-meteo.readthedocs.io/en/latest/).


## Contributing
ProMeteo is an open project. Suggestions, corrections, and contributions are very welcome!

- Open an issue for bugs or feature requests
- Submit a pull request to contribute code or improvements
- Or contact me directly via email (see GitHub profile page)

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
