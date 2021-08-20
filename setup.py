import subprocess
import os,sys
from setuptools import setup,find_packages

requirements = ["torch", "torchvision"]
nvcc = subprocess.check_output("nvcc --version|grep cuda", shell=True)
nvcc = nvcc.decode('utf-8')

is_conda = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
if 'cuda_11.1' in nvcc:
    os.system("pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html")

elif 'cuda_10.2' in nvcc:
    cuda_version = '10.2'
else:
    raise EnvironmentError("Please install CUDA 11.1 or 10.2 on your machine")


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
