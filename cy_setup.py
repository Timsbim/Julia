# distutils-imports
from distutils.core import setup
from distutils.extension import Extension

# Cython-imports
from Cython.Build import cythonize

# Third-party-imports
import numpy as np

# Define extension modules
ext_modules = [
    Extension(
        "cy_julia",
        ["cy_julia.pyx"]
    )
]

# Setup call
setup(
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={"language_level": 3}
    ),
    include_dirs=[np.get_include()]
)

"""
The -openmp flags
- are not recognized and ignored on Windows 10 (same for -fopenmp)
- disrupt the process on MacOS (not so-output generated)

Old version:
    Extension(
        "cy_julia",
        ["cy_julia.pyx"],
        extra_compile_args=["-openmp"],
        extra_link_args=["-openmp"]
    )
"""