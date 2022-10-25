import scipy.io as sio
import matplotlib.pyplot as plt
from evaluation.RasterizeLine import drawfn
import numpy as np
from evaluation.prmeter import PrecisionRecallMeter
import glob
import os.path as osp
from tqdm import tqdm
import multiprocessing
import json
import argparse
from scipy import interpolate
class LSDEvaluator(object):
    def __init__(self, pred_json, gt_json, thresholds, cmp='g', height=0,width=0):
        self.pred_json = pred_json
        self.gt_json = gt_json
        self.thresholds = thresholds
        self.meter = PrecisionRecallMeter(self.thresholds, cmp=cmp)
        self.height = height
        self.width = width
        with open(self.pred_json,'r') as f:
            self.pred_dict = json.load(f)
        
        with open(self.gt_json,'r') as f:
            self.gt_dict = json.load(f)

        self.pred_by_name = {} 
        self.gt_by_name = {}

        for pred in self.pred_dict:
            self.pred_by_name[pred['filename']] = pred

        for gt in self.gt_dict:
            self.gt_by_name[gt['filename']] = gt
        
        self.filenames = sorted(self.gt_by_name.keys())

        for pred_name in sorted(self.pred_by_name.keys()):
            assert pred_name in self.filenames

            
        
    def eval_for_image(self, index):
        fname = self.filenames[index]
        gt = self.gt_by_name[fname]
        pred = self.pred_by_name[fname]

        gt_lines = np.array(gt['lines'],dtype=np.float32)
        pred_lines = np.array(pred['lines_pred'],dtype=np.float32)
        pred_scores = np.array(pred['lines_score'],dtype=np.float32)
        pred_lines = np.concatenate((pred_lines,pred_scores[:,None]),axis=-1)
        
        width = gt['width']
        height = gt['height']

        pw = pred['width']
        ph = pred['height']

        if pw!=width or ph!=height:
            sx = float(width/pw)
            sy = float(height/ph)
            scale = np.array([sx,
                              sy,
                              sx,
                              sy],dtype=np.float32).reshape(-1,4)
            pred_lines[:,:4] = pred_lines[:,:4]*scale
            
        if self.height > 0 and self.width > 0:
            sx = float(self.width/height)
            sy = float(self.height/height)
            scale = np.array([sx,
                              sy,
                              sx,
                              sy],dtype=np.float32)

            scale = scale.reshape((1,4))
            gt_lines *= scale
            pred_lines[:,:4] = pred_lines[:,:4]*scale
            return self.meter(pred_lines,gt_lines,self.height,self.width)
        else:
            return self.meter(pred_lines,gt_lines,gt['height'],gt['width'])

    def __call__(self, num_workers=16, per_image = True):
        # self.eval_for_image(0)
        if num_workers>0:
            with multiprocessing.Pool(num_workers) as p:
                self.results = results = list(tqdm(p.imap(self.eval_for_image,
                                                      range(len(self.filenames))), total=len(self.filenames)))
        else:
            self.results = [self.eval_for_image(i) for i in range(len(self.filenames))]

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

if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument('--pred',type=str,required=True,help='the json file for the wireframe or line segment predictions')
    parser.add_argument('--benchmark', type=str, choices = ['wireframe','york'], required=True)
    #parser.add_argument('--thresholds', type=list, default = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.97,0.99,0.995, 0.997,0.999])
    parser.add_argument('--thresholds', type=float, nargs='+', default = None)
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--nthreads', default=16, type=int)
    parser.add_argument('--cmp', default='g', choices=['g','l'])
    
    args = parser.parse_args()
    
    if args.thresholds is None:
        args.thresholds = [0.01*i for i in range(100)]
    
    if args.benchmark == 'wireframe':
        args.gt = 'data/wireframe/test.json'
    elif args.benchmark == 'york':
        args.gt = 'data/york/test.json'

    evaluator = LSDEvaluator(
        args.pred,
        args.gt,
        args.thresholds,
        args.cmp,
    )
    # evaluator.eval_for_image(32,0)
    results = evaluator(args.nthreads,per_image=False)
    rcs = sorted(results['avg_recall'])
    prs = sorted(results['avg_precision'])[::-1]
    print((2 * np.array(prs) * np.array(rcs) / (np.array(prs) + np.array(rcs))))
    print(evaluator.thresholds)
    print(
        "f measure is: ",
        (2 * np.array(prs) * np.array(rcs) / (np.array(prs) + np.array(rcs))).max(),
    )
    recall = np.concatenate(([0.0], rcs, [1.0]))
    precision = np.concatenate(([0.0], prs,[ 0.0]))

    for i in range(precision.size-1, 0, -1): 
        precision[i-1] = max(precision[i-1], precision[i])

    i = np.where(recall[1:] != recall[:-1])[0]
    AP = sum((recall[i+1]-recall[i])*precision[i+1])
    print("AP is: ", np.sum((recall[i + 1] - recall[i]) * precision[i + 1]))

    # plt.plot(recall,precision)
    # plt.show()
    f = interpolate.interp1d(rcs,prs, kind='linear', bounds_error=False)
    x = np.arange(rcs[0],rcs[-1],0.01)
    y = f(x)

    label = args.label
    evaluation_dict = {'recall': x,'precision': y,'f': (2 * np.array(prs) * np.array(rcs) / (np.array(prs) + np.array(rcs))).max(),'AP': AP,'label': label,}

    output_path = '{}-aph.mat'.format(args.pred[:-5])
    sio.savemat(output_path,evaluation_dict)

    # mdict = {
    #     'recall': x,
    #     'precision': y,
    #     'f':  (2 * np.array(prs) * np.array(rcs) / (np.array(prs) + np.array(rcs))).max(),
    #     'AP': AP,
    #     'label': label
    # }
    # import pdb; pdb.set_trace()
    # parser = argparse.ArgumentParser("program of evaluating heat map based F measure and AP metric")
    # parser.add_argument('--path',type=str,required=True)
    # parser.add_argument('--label', type=str, required=True)
    # parser.add_argument('--thresholds', type=float, nargs='+', default=[0.0,0.5,0.6,0.7,0.8,0.9,0.95,0.97,0.99,0.995, 0.997,0.999])
    # # parser.add_argument('--descending', action='store_true')
    # parser.add_argument('--cmp',type=str,default='g')


