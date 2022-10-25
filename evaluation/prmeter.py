from .EdgeEval import correspond
from .RasterizeLine import drawfn
import numpy as np

# cmp_dict = {
#     'g': lambda a,b: a>b,
#     'l': lambda a,b: a==b,
#     'e': lambda a,b: a<b,
# }
class PrecisionRecallMeter(object):
    def __init__(self, thresholds, maxDist=0.01, cmp = 'g'):
        assert isinstance(thresholds,(list, tuple))

        self.thresholds = thresholds
        self.maxDist = maxDist
        # self.descending = descending
        if cmp == 'g':
            self.cmp = self.cmp_g
        elif cmp == 'e':
            self.cmp = self.cmp_e
        elif cmp == 'l':
            self.cmp = self.cmp_l
        else:
            raise  NotImplementedError()

    @staticmethod
    def cmp_g(a,b):
        return a>b

    @staticmethod
    def cmp_e(a,b):
        return a==b

    @staticmethod
    def cmp_l(a,b):
        return a<b
    def __call__(self, pred, gt, height, width):

        gt = np.ascontiguousarray(gt)
        gt_map = drawfn(gt,height,width)
        recalls = np.array([0]*len(self.thresholds),dtype=np.float32)
        precisions = np.array([0]*len(self.thresholds),dtype=np.float32)
        cntR = np.array([0]*len(self.thresholds),dtype=np.float32)
        sumR = np.array([0]*len(self.thresholds),dtype=np.float32)
        cntP = np.array([0]*len(self.thresholds),dtype=np.float32)
        sumP = np.array([0]*len(self.thresholds),dtype=np.float32)

        tp = np.array([0]*len(self.thresholds),dtype=np.float32)
        fp = np.array([0]*len(self.thresholds),dtype=np.float32)
        gt = np.array([0]*len(self.thresholds),dtype=np.float32)

        for i, t in enumerate(self.thresholds):
            idx = np.where(self.cmp(pred[:,4],t))[0]
            # if self.descending:
            #     idx = np.where(pred[:,4]<t)[0]
            # else:
            #     idx = np.where(pred[:,4]>t)[0]
            pred_t = np.ascontiguousarray(pred[idx,:4])
            pred_map = drawfn(pred_t,height,width)

            matchE, matchG, _ = correspond(pred_map, gt_map,self.maxDist)
            cntR[i] = np.sum(matchG>0)
            sumR[i] = np.sum(gt_map>0)
            cntP[i] = np.sum(matchE>0)
            sumP[i] = np.sum(pred_map>0)

            matchE = np.array(matchE>0,dtype=np.float32)

            tp[i] = matchE.sum()
            fp[i] = np.sum(pred_map) - matchE.sum()
            gt[i] = gt_map.sum()
            #fp: sumP-cntR
            #gt: sumR
            #tp: cntP

            recalls[i] = cntR[i] / (sumR[i]+1e-15)
            precisions[i] = cntP[i] / (sumP[i]+1e-15)



        fscore = 2*recalls*precisions/(recalls+precisions+1e-6)

        return {'p':precisions, 'r':recalls, 'f':fscore,
                'tp': tp,
                'fp': fp,
                'gt': gt,
                'sumR': sumR,
                'cntR': cntR,
                'sumP': sumP,
                'cntP': cntP}
