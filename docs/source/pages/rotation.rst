Rotation
========

 - **LEC (Local Earth Coordinate)**:
    - ``u``: east, ``v``: north, ``w``: vertical.
    - Requires instrument model and azimuth.
    - Supported models:
      - RM YOUNG 81000: https://www.youngusa.com/products/3d-sonic-anemometer-model-81000/
      - Campbell CSAT3: https://www.campbellsci.com/csat3

  - **Streamline**:
    - Reference rotates dynamically with the mean wind.
    - ``u``: streamwise, ``v``: crosswise, ``w``: vertical to streamline plane.
    - Does **not** require azimuth.

- Wind direction is computed in the **LEC frame** using the meteorological convention (direction *from* which wind blows, relative to North).
