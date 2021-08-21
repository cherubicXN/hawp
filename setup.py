import subprocess
import os,sys
from setuptools import setup,find_packages

requirements = ["torch", "torchvision"]

if __name__ == '__main__':

    setup(
        name="hawp",
        version="0.1.1",
        author="Nan Xue",
        description="Holistically-Attracted Wireframe Parsing (in pytorch)",
        packages=find_packages(),
        python_requires='>=3.6',
        install_requires = [
            "yacs", 
            "opencv-python",
            "matplotlib",
            "scikit-image",
            "tqdm",
            "torch",
            "torchvision",
        ]
    )
