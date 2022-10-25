import scipy.io as sio
import matplotlib.pyplot as plt
from evaluation.RasterizeLine import drawfn
import numpy as np

if __name__ == '__main__':
    mat = sio.loadmat('../outputs/afm_box_b4/wireframe_afm/00037266.mat')
    gt = mat['gt']
    pred = mat['pred']
    height = mat['height']
    width = mat['width']
    G = drawfn(np.ascontiguousarray(gt),height,width)
    P = drawfn(np.ascontiguousarray(pred[:,:4]), height,width)
    import pdb
    pdb.set_trace()
