from multiprocessing import Pool
from array import array
from pathlib import Path
from time import perf_counter

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
    x_left, x_right = x_interval
    y_bottom, y_top = y_interval

    x_res, y_res = set_resolutions(x_interval, y_interval, resolution)

    x_axis = np.linspace(x_left, x_right, x_res, endpoint=True)
    y_axis = np.linspace(y_top, y_bottom, y_res, endpoint=True)

    return sum(np.meshgrid(x_axis, 1.j * y_axis)).flatten()


def julia(c, z):
    n = 0
    while abs(z) < 2. and n < 300:
        z = z * z + c
        n += 1
    return n


if __name__ == '__main__':

    start = perf_counter()

    x_interval = (-1.5, 1.5)
    y_interval = (-.95, .95)
    resolution = 2000

    c = -0.62772 - 0.42193j
    args = (
        (c, arg) for arg in prepare_data(x_interval, y_interval, resolution)
    )
    with Pool(12) as p:
        flat_results = p.starmap(julia, args)

    end = perf_counter()
    print(f"Duration of calculations: {end - start:.2f} seconds")

    x_res, y_res = set_resolutions(x_interval, y_interval, resolution)
    pil_show(flat_results, x_res, y_res, file="julia_1.png")
    plt.imsave(
        f"julia_2.png",
        np.array(flat_results).reshape((y_res, x_res)),
        cmap="binary"
    )
