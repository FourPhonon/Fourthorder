#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy
from distutils.core import setup
from distutils.extension import Extension

# Add the location of the "spglib/spglib.h" to this list if necessary.
# Example: INCLUDE_DIRS=["/home/user/local/include"]
INCLUDE_DIRS=["/opt/software/spglib.1.6.4/include"]
# Add the location of the spglib shared library to this list if necessary.
# Example: LIBRARY_DIRS=["/home/user/local/lib"]
LIBRARY_DIRS=["/opt/software/spglib.1.6.4/lib"]


# Set USE_CYTHON to True if you want include the cythonization in your build process.
USE_CYTHON=False

ext=".pyx" if USE_CYTHON else ".c"

extensions=[Extension("Fourthorder_core",
                      ["Fourthorder_core"+ext],
                      include_dirs=[numpy.get_include()]+INCLUDE_DIRS,
                      library_dirs=LIBRARY_DIRS,
                      runtime_library_dirs=LIBRARY_DIRS,
                      libraries=["symspg"])]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions=cythonize(extensions)
setup(
    name="Fourthorder",
    ext_modules=extensions
    )
