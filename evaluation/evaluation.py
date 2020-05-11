import numpy as np
import scipy.io as sio
import glob
import os
import os.path as osp
from .prmeter import PrecisionRecallMeter
import multiprocessing
from tqdm import tqdm

class LSDEvaluator(object):
    def __init__(self, root, thresholds, cmp='g', height=0,width=0):
        self.root = root
        self.filenames = glob.glob(osp.join(root,'*.mat'))
        self.thresholds = thresholds
        self.meter = PrecisionRecallMeter(self.thresholds,cmp=cmp)
        self.height =height
        self.width  = width

    def eval_for_image(self, index):
        mat = sio.loadmat(self.filenames[index])
        gt = mat['gt']
        pred = mat['pred']
        height = mat['height'].item()
        width = mat['width'].item()
        # import pdb
        # pdb.set_trace()
        if self.height>0 and self.width>0:
            sx = float(self.width/width)
            sy = float(self.height/height)
            scale = np.array([sx,
                              sy,
                              sx,
                              sy],dtype=np.float32)
            scale = scale.reshape((1,4))
            gt*=scale
            pred[:,:4] = pred[:,:4]*scale
            return self.meter(pred,gt,self.height,self.width)
        else:
            return self.meter(pred, gt, height, width)

    def __call__(self, num_workers=16, per_image = True):
        # self.eval_for_image(0)
        with multiprocessing.Pool(num_workers) as p:
            self.results = results = list(tqdm(p.imap(self.eval_for_image,
                                                      range(len(self.filenames))), total=len(self.filenames)))
        if per_image:
            self.precisions = np.concatenate([r['p'][:,None] for r in self.results],axis=1)
            self.recalls = np.concatenate([r['r'][:, None] for r in self.results], axis=1)

            self.average_precisions = np.mean(self.precisions,axis=1)
            self.average_recalls = np.mean(self.recalls, axis=1)
            self.fmeasure = 2*self.average_precisions*self.average_recalls/(self.average_recalls+self.average_precisions)

            return {'precisions':self.precisions, 'recalls': self.recalls,
                    'avg_precision':self.average_precisions,
                    'avg_recall': self.average_recalls,
                    'avg_fmeasure': self.fmeasure,
                    'filenames': self.filenames}
        else:
            sumtp = sum(res['tp'] for res in results)
            sumfp = sum(res['fp'] for res in results)
            sumgt = sum(res['gt'] for res in results)
            # import pdb
            # pdb.set_trace()
            # rcs = sorted(sumtp/sumgt)
            # prs = sorted(sumtp/np.maximum(sumtp+sumfp,1e-9))[::-1]
            rcs = sumtp/sumgt
            prs = sumtp/np.maximum(sumtp+sumfp,1e-9)
            # temp = np.concatenate(([0],prs))
            # idx = np.where((temp[1:]-temp[:-1])>0)[0]
            # rcs = rcs[idx]
            # prs = prs[idx]

            return {'avg_precision': prs, 'avg_recall':rcs}

