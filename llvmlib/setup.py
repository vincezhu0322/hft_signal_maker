from distutils.core import setup

from hft_signal_maker.llvmlib.numba_util import export_helper

setup(ext_modules=[export_helper.distutils_extension()])
