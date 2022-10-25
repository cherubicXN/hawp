import scipy.io as sio
import matplotlib.pyplot as plt
from evaluation.RasterizeLine import drawfn
import numpy as np
from evaluation.prmeter import PrecisionRecallMeter
import glob
import os.path as osp
from tqdm import tqdm
import multiprocessing
if __name__ == '__main__':

    files = glob.glob('../outputs/afm_box_b4/wireframe_afm/*.mat')
    meter = PrecisionRecallMeter([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # results = [None]*len(files)
    def eval_on_image(i):
        mat = sio.loadmat(files[i])
        gt = mat['gt']
        pred = mat['pred']
        height = mat['height']
        width = mat['width']

        return meter(pred,gt,height,width)

    # for i in range(len(files)):
    #     eval_on_image(i)
    #     import pdb
    #     pdb.set_trace()
    with multiprocessing.Pool(32) as p:
        results = list(tqdm(p.imap(eval_on_image, range(len(files))), total=len(files)))
    precisions = np.concatenate([r['p'][:,None] for r in results],axis=1)
    recalls = np.concatenate([r['r'][:,None] for r in results],axis=1)
    
    import pdb
    pdb.set_trace()

        # mat = sio.loadmat('../outputs/afm_box_b4/wireframe_afm/00031546.mat')

