import numpy as np
import scipy.io as sio
import os
import os.path as osp
import argparse
import json
from tqdm import tqdm
OLD_ROOT = '/home/nxue2/dwp/result/%s_0.5_0.5'
GT_ANNO = './epnet/data/%s/test.json'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',type=str,required=True)
    parser.add_argument('--out',type=str,required=True)

    args = parser.parse_args()
    dataset = args.dataset
    with open(GT_ANNO%dataset,'r') as buffer:
        annotations = json.load(buffer)

    old_dir = OLD_ROOT%dataset

    thresholds_str = os.listdir(old_dir)
    assert  all([osp.isdir(osp.join(old_dir,t)) for t in thresholds_str])
    thresholds = sorted([int(t) for t in thresholds_str])
    # thresholds = [2, 6, 10, 20, 30, 50]

    out_dir = args.out
    os.makedirs(out_dir,exist_ok=True)

    for i, anno in enumerate(tqdm(annotations)):
        lines = []
        fname = anno['filename'].rstrip('.png')
        for t in thresholds:
            mat = sio.loadmat(osp.join(old_dir,str(t),fname+'.mat'))
            lines_t = np.array(mat['lines'],dtype=np.float32)
            if lines_t.shape[0] == 0:
                continue

            scores = np.ones((lines_t.shape[0],1),dtype=np.float32)*t

            lines_t = np.concatenate((lines_t,scores),axis=1)
            # assert lines_t.shape[1] == 5
            lines.append(lines_t)
        # if i==371:
        #     import pdb
        #     pdb.set_trace()
        lines = np.concatenate(lines,axis=0)
        gt = np.array(anno['lines'],dtype=np.float32)

        height = anno['height']
        width  = anno['width']
        mdict = {
            'pred': lines,
            'gt':  gt,
            'height': height,
            'width': width
        }
        sio.savemat(osp.join(out_dir,fname+'.mat'),mdict)
