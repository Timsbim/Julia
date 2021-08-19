from itertools import product
from pathlib import Path
from time import strftime, perf_counter
from array import array

from PIL import Image


def pil_show(flat_image, x_res, y_res, file):
    """Idea from: https://github.com/mynameisfiber/high_performance_python_2e
    """
    scale_factor = float(max(flat_image))
    output = array(
        'B', (int(value / scale_factor * 255) for value in flat_image)
    )
    img = Image.new("L", (x_res, y_res))
    img.frombytes(output.tobytes(), "raw", "L", 0, -1)
    img.save(file)


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
    delta = (x_right - x_left) / x_res
    x_axis = [x_left + delta * i for i in range(x_res)]
    delta = (y_top - y_bottom) / y_res
    y_axis = [y_bottom + delta * i for i in range(y_res)]

    return tuple(x + 1.j * y for y in y_axis for x in x_axis)


def julia(c, z, max_iter):
    """Obvious ... :))"""
    n = 0
    while (z.real * z.real + z.imag * z.imag < 4.) and n < max_iter:
        z = z * z + c
        n += 1
    return n


if __name__ == '__main__':

    # Image path (create, if it doesn't exist)
    path = Path().cwd() / "Images" / "Test"
    path.mkdir(exist_ok=True)

    # Relatively good section (not for all) and resolution
    x_interval = (-1.6, 1.6)
    y_interval = (-1., 1.)
    resolution = 1000
    x_res, y_res = set_resolutions(x_interval, y_interval, resolution)
    max_iter = 500

    # List of interesting c constants
    c_list = [
        -0.62772 - 0.42193j,
        -0.74543 + 0.11301j,
        -0.75 + 0.11j,
        -0.1 + 0.651j,
        -0.8 + 0.156j
    ]
    zs = prepare_z(x_interval, y_interval, resolution)
    for i, c in enumerate(c_list, start=1):
        print(f"{strftime('%H:%M:%S')}: Calculating {i}. julia set ...")
        start = perf_counter()
        # Calculating image data using a multiprocessing pool
        flat_image = [julia(c, z, max_iter) for z in zs]
        end = perf_counter()
        print(f"{strftime('%H:%M:%S')}: ... done (in {end - start:.2f} secs.)")

        print(f"{strftime('%H:%M:%S')}: Saving julia_({c:.5f}) ...")
        pil_show(flat_image, x_res, y_res, file=path / f"julia_{c:.5f}.png")
