from itertools import product
from pathlib import Path
from time import strftime, perf_counter
from array import array

import numpy as np
from matplotlib import pyplot as plt

from cy_julia import calc_julia, calc_julia_nmp


def set_resolutions(x_interval, y_interval, resolution):
    """Create evenly distributed resolutions:
     - Use given resolution for the larger section
     - Adjust given resolution proportionally for shorter section
     - Minor adjustments: Always use odd resolutions
    """
    x_left, x_right = x_interval
    y_bottom, y_top = y_interval

    if not resolution % 2:
        resolution += 1
    x_to_y_ratio = (x_right - x_left) / (y_top - y_bottom)
    if x_to_y_ratio >= 1:
        x_res = resolution
        y_res = int(resolution / x_to_y_ratio)
        if not y_res % 2:
            y_res += 1
    else:
        x_res = int(resolution * x_to_y_ratio)
        if not x_res % 2:
            x_res += 1
        y_res = resolution

    return x_res, y_res


def prepare_z(x_interval, y_interval, resolution):
    """Prepare z-data grid (flattened) for actual calculation:
     - Rectangle defined by top-left point (x_interval[0], y_interval[1]) and
       bottom right point (x_interval[1], y_interval[0])
     - Grid spacing is given by the resolution
    """
    x_left, x_right = x_interval
    y_bottom, y_top = y_interval
    x_res, y_res = set_resolutions(x_interval, y_interval, resolution)
    x_axis = np.linspace(x_left, x_right, x_res, endpoint=True)
    y_axis = np.linspace(y_top, y_bottom, y_res, endpoint=True)

    return sum(np.meshgrid(x_axis, 1.j * y_axis)).flatten()


# Image path (create, if it doesn't exist)
path = Path().cwd() / "Images" / "Test"
path.mkdir(exist_ok=True)

# Some Matplotlib color maps
cmaps = [
    "binary", "Blues",
    "seismic", "plasma", "viridis",
    "Spectral", "coolwarm",
    "jet"
]

# Relatively good section (not for all) and resolution
x_interval = (-1.4, 1.4)
y_interval = (-1.25, 1.25)
resolution = 1000
x_res, y_res = set_resolutions(x_interval, y_interval, resolution)
zs = prepare_z(x_interval, y_interval, resolution)
max_iter = 500

# Lists of interesting c constants
"""
c_list = [
    -0.62772 - 0.42193j,
    -0.74543 + 0.11301j,
    -0.75 + 0.11j,
    -0.1 + 0.651j,
    -0.8 + 0.156j
]
c_list = prepare_z((-0.8, 0.8), (-0.8, 0.8), 60)
c_list = -0.4 + np.linspace(0.583, 0.61, 28, endpoint=True) * 1.j
c_list = -0.507 - np.linspace(0.513, 0.525, 25, endpoint=True) * 1.j
c_list = -0.773 + np.linspace(0.105, 0.133, 29, endpoint=True) * 1.j
c_list = 0.373 - np.linspace(0.132, 0.176, 45, endpoint=True) * 1.j
c_list = 0.373 - np.linspace(0.270, 0.290, 20, endpoint=True) * 1.j
"""
c_list = -0.100 + np.linspace(0.650, 0.653, 31, endpoint=True) * 1.j
for i, c in enumerate(c_list, start=1):
    print(f"{strftime('%H:%M:%S')}: Calculating {i}. julia set ...")
    start = perf_counter()
    img_data = calc_julia(c, zs, max_iter).reshape((y_res, x_res))
    end = perf_counter()
    print(f"{strftime('%H:%M:%S')}: ... done (in {end - start:.2f} secs.)")

    # Saving one image per color map
    for j, cmap in enumerate(cmaps[:3], start=1):
        print(f"{strftime('%H:%M:%S')}: Saving julia_({c:.4f})-{j}.png ...")
        plt.imsave(path / f"julia_({c:.4f})-{j}.png", img_data, cmap=cmap)
