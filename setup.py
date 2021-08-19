import glob
import os

import torch
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

requirements = ["torch", "torchvision"]


setup(
    name="hawp",
    version="0.1.1",
    author="Nan Xue",
    description="holistically-attracted wireframe parsing (in pytorch)",
    packages=find_packages(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
