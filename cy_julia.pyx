def calc_julia(
    double complex c,
    zs,
    unsigned int max_iter
):
    cdef unsigned int i, n
    cdef double complex z

    result = [0] * len(zs)
    for i, z in enumerate(zs):
        n = 0
        while (
            (z.real * z.real + z.imag * z.imag < 4.)
            and (n < max_iter)
        ):
            z = z * z + c
            n += 1
        result[i] = n

    return result
