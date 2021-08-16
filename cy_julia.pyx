#cython: boundscheck=False
from cython.parallel import prange
import numpy as np
cimport numpy as np


def calc_julia(
    double complex c,
    double complex[:] zs,
    int max_iter
):
    cdef long i, length
    cdef double complex z
    cdef int[:] result = np.empty(len(zs), dtype=np.int32)

    length = len(zs)
    with nogil:
        for i in prange(length, schedule="guided"):
            z = zs[i]
            result[i] = 0
            while (
                (z.real * z.real + z.imag * z.imag < 4.)
                and (result[i] < max_iter)
            ):
                z = z * z + c
                result[i] += 1

    return np.array(result, dtype=np.int32)


def calc_julia_nmp(
    double complex c,
    double complex[:] zs,
    int max_iter
):
    cdef int i, n
    cdef double complex z
    cdef int[:] result = np.empty(len(zs), dtype=np.int32)

    for i in range(len(zs)):
        z = zs[i]
        n = 0
        while (
            (z.real * z.real + z.imag * z.imag < 4.)
            and (n < max_iter)
        ):
            z = z * z + c
            n += 1
        result[i] = n

    return np.array(result, dtype=np.int32)
