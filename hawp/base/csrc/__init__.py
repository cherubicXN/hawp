from torch.utils.cpp_extension import load 
import glob
import os.path as osp

__this__ = osp.dirname(__file__)

_C = load(name='_C',sources=[
    osp.join(__this__,'binding.cpp'),
    osp.join(__this__,'linesegment.cu'),
]
)

__all__ = ["_C"]

#_C = load(name='base._C', sources=['lltm_cuda.cpp', 'lltm_cuda_kernel.cu'])
