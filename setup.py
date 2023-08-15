import glob
import os

from setuptools import find_packages
from setuptools import setup

setup(
    name="hawp",
    version="1.0",
    author="nxue",
    description="Holistically-Attracted Wireframe Parsing",
    packages=find_packages(),
    install_requires=[
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
        "seaborn",
        "easydict",
    ],
    extras_require={
        "dev": [
            "pycolmap",
        ]
    }
)
