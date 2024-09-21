from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os


ext = Extension(
    name="Zalatel_import",
    sources=["Zalatel_import.pyx"],
    include_dirs=[np.get_include()],
)

setup(
    name="YourProjectName",
    ext_modules=cythonize([ext]),
    zip_safe=False,
)