import os
from setuptools import setup
from Cython.Distutils import build_ext
from distutils.extension import Extension
import subprocess
import numpy as np
import glob

try:
    NP_INCLUDE = np.get_include()
except AttributeError:
    NP_INCLUDE = np.get_numpy_include()

CSAPP_INCLUDE = os.path.abspath('CSA++')

INCLUDE = [os.path.abspath('include'),
            CSAPP_INCLUDE,
            NP_INCLUDE,
            os.path.abspath('Util')]
# SOURCES = ['CSA++/csa.cc','match.cc','correspondPixels.cc']
SOURCES = ['EdgeMapEval.cc','CSA++/csa.cc','match.cc','correspondPixels.pyx'] + glob.glob('Util/*.cc')
# import pdb
# pdb.set_trace()

class custom_build_ext(build_ext):
    def build_extensions(self):
        build_ext.build_extensions(self)

ext_modules = [
    Extension("EdgeMapEval",
        sources=SOURCES,
        include_dirs=INCLUDE,
        language='c++',
        extra_compile_args=["-std=c++11"],
        )
]

setup(
    ext_modules = ext_modules,
    cmdclass = {
        'build_ext': custom_build_ext
    }
)