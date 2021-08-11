from multiprocessing import Pool
from array import array
from pathlib import Path
from time import strftime

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def pil_show(flat, x_res, y_res, file=None):
    scale_factor = float(max(flat))
    output = array(
        'B', (int(value / scale_factor * 255) for value in flat)
    )
    img = Image.new("L", (x_res, y_res))
    img.frombytes(output.tobytes(), "raw", "L", 0, -1)
    img.show()
    if file is not None:
        path = Path().cwd() / file
        img.save(path)


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
        x_res = int(resolution / x_to_y_ratio)
        if not x_res % 2:
            x_res += 1
        y_res = resolution

    return x_res, y_res


def prepare_data(x_interval, y_interval, resolution):
    """Prepare base data grid (flattened) for actual calculation:
     - Rectangle defined by top-left point (x_interval[0], y_interval[1]) and
       bottom right point (x_interval[1], y_interval[0])
     - Grid spacing is given through the resolution
    """
    x_left, x_right = x_interval
    y_bottom, y_top = y_interval

    x_res, y_res = set_resolutions(x_interval, y_interval, resolution)

    x_axis = np.linspace(x_left, x_right, x_res, endpoint=True)
    y_axis = np.linspace(y_top, y_bottom, y_res, endpoint=True)

    return sum(np.meshgrid(x_axis, 1.j * y_axis)).flatten()


def julia(c, z):
    """Obvious ... :)"""
    n = 0
    while abs(z) < 2. and n < 300:
        z = z * z + c
        n += 1
    return n


if __name__ == '__main__':

    # Matplotlib color maps
    cmaps = ["binary", "Blues", "seismic"]

    # Relatively good section (not for all) and resolution
    x_interval = (-1.6, 1.6)
    y_interval = (-1., 1.)
    resolution = 1000
    x_res, y_res = set_resolutions(x_interval, y_interval, resolution)

    # List of interesting c's
    c_list = [
        -0.62772 - 0.42193j,
        -0.74543 + 0.11301j,
        -0.75 + 0.11j,
        -0.1 + 0.651j,
        -0.8 + 0.156j
    ]
    for i, c in enumerate(c_list, start=1):
        print(f"{strftime('%H:%M:%S')}: Calculating {i}. set ...")
        args = (
            (c, arg)
            for arg in prepare_data(x_interval, y_interval, resolution)
        )
        # Calculating image data using a multiprocessing pool
        with Pool(12) as p:
            img_data = np.array(p.starmap(julia, args)).reshape((y_res, x_res))

        # Saving image using several color maps
        for j, cmap in enumerate(cmaps, start=1):
            print(f"{strftime('%H:%M:%S')}: Saving julia_{i}-{j}.png ...")
            plt.imsave(f"julia_{i}-{j}.png", img_data, cmap=cmap)
