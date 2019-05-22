from distutils.core import setup
from Cython.Build import cythonize

setup(name='FUDD',
      ext_modules=cythonize("FastUnivariateDensityDerivative.pyx"))