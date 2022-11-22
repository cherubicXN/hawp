import glob
import os

import torch
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

requirements = [
        "torch", 
        "torchvision",
        "opencv-python",
        "cython",
        "matplotlib",
        "yacs",
        "scikit-image",
        "tqdm",
        "python-json-logger",
        "h5py",
        "shapely",
        "pycolmap",
        "seaborn",
        ]


setup(
    name="hawp",
    version="1.0",
    author="nxue",
    description="Holistically-Attracted Wireframe Parsing",
    packages=find_packages(['hawp']),
    install_requires=requirements,
)
