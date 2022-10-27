import torch

from hawp.base.utils.comm import to_device
from hawp.base.utils.logger import setup_logger
from hawp.base.utils.checkpoint import DetectronCheckpointer
from hawp.base.utils.metric_evaluation import TPFP, AP

from hawp.fsl.config import cfg
from hawp.fsl.config.paths_catalog import DatasetCatalog
from hawp.fsl.dataset import build_test_dataset
from hawp.fsl.model.build import build_model

import os
import os.path as osp
import argparse
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import numpy as np
import importlib
import time
import random

AVAILABLE_DATASETS = {
    'wireframe': 'wireframe_test', 
    'york': 'york_test'
}
THRESHOLDS = [5, 10, 15]

def plot_pr_curve(P, R, F, path):
    f_scores = np.linspace(0.2,0.9,num=8).tolist()
            
    for f_score in f_scores:
        x = np.linspace(0.01,1)
        y = f_score*x/(2*x-f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color=[0,0.5,0], alpha=0.3)
        plt.annotate("f={0:0.1}".format(f_score), xy=(0.9, y[45] + 0.02), alpha=0.4,fontsize=10)
    
    plt.rc('legend',fontsize=10)
    plt.grid(True)
    plt.axis([0.0, 1.0, 0.0, 1.0])
    plt.xticks(np.arange(0, 1.0, step=0.1))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.yticks(np.arange(0, 1.0, step=0.1))
    plt.plot(rcs,pcs,'r-')
    plt.plot(R,P,'.',color=[0,0.5,0],)
    plt.annotate("f={0:0.3}".format(F), xy=(R, P + 0.02), alpha=0.4,fontsize=10)

    plt.title(sAP_string)
    plt.savefig(path)

def sAPEval(result_list, annotations_dict, threshold):
    tp_list, fp_list, scores_list = [],[],[]
    n_gt = 0
    for res in result_list:
        filename = res['filename']
        gt = annotations_dict[filename]
        lines_pred = np.array(res['lines_pred'],dtype=np.float32)
        scores = np.array(res['lines_score'],dtype=np.float32)
        sort_idx = np.argsort(-scores)
        
        lines_pred = lines_pred[sort_idx]
        scores = scores[sort_idx]
        # import pdb; pdb.set_trace()
        lines_pred[:,0] *= 128/float(res['width'])
        lines_pred[:,1] *= 128/float(res['height'])
        lines_pred[:,2] *= 128/float(res['width'])
        lines_pred[:,3] *= 128/float(res['height'])

        lines_gt   = np.array(gt['lines'],dtype=np.float32)
        lines_gt[:,0]  *= 128/float(gt['width'])
        lines_gt[:,1]  *= 128/float(gt['height'])
        lines_gt[:,2]  *= 128/float(gt['width'])
        lines_gt[:,3]  *= 128/float(gt['height'])
        
        assert gt['width'] == res['width'] and gt['height'] == res['height']
        
        tp, fp = TPFP(lines_pred,lines_gt,threshold)
        n_gt += lines_gt.shape[0]
        tp_list.append(tp)
        fp_list.append(fp)
        scores_list.append(scores)

    tp_list = np.concatenate(tp_list)
    fp_list = np.concatenate(fp_list)
    scores_list = np.concatenate(scores_list)
    idx = np.argsort(scores_list)[::-1]
    tp = np.cumsum(tp_list[idx])/n_gt
    fp = np.cumsum(fp_list[idx])/n_gt
    rcs = tp
    pcs = tp/np.maximum(tp+fp,1e-9)
    F_list = (2*rcs*pcs/(rcs+pcs+1e-9))
    F_list = np.nan_to_num(F_list, 0)
    F = F_list.max()
    
    P = pcs[F_list.argmax()]
    R = rcs[F_list.argmax()]

    sAP = AP(tp,fp)

    return sAP, P, R, F
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HAWP Testing')
    parser.add_argument('config', help = 'the path of config file')
    parser.add_argument("--ckpt",type=str,required=True)

    parser.add_argument("--dataset", default='wireframe', choices=['wireframe','york'])

    parser.add_argument("--j2l", default = None, type = float, help = 'the threshold for junction-line attraction')
    parser.add_argument("--rscale",default=2, type=int, help='the residual scale')

    parser.add_argument('--seed',default=42,type=int)
    parser.add_argument('--output', default=None, help = 'the path of outputs')

    args = parser.parse_args()

    config_path = args.config
    cfg.merge_from_file(config_path)
    root = args.output

    if root is None:
        root = os.path.dirname(args.ckpt)

    cfg.DATASETS.TEST = (AVAILABLE_DATASETS.get(args.dataset),)

    logger = setup_logger('hawp.testing', root)
    logger.info(args)
    logger.info("Loaded configuration file {}".format(config_path))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = cfg.MODEL.DEVICE

    model = build_model(cfg)
    model = model.to(cfg.MODEL.DEVICE)

    if args.rscale is not None:
        model.use_residual = args.rscale

    if args.j2l:
        model.j2l_threshold = args.j2l
    
    if args.dataset == 'york':
        model.topk_junctions = 512
    
    test_datasets = build_test_dataset(cfg)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'])

    model = model.eval()

    for name, dataset in test_datasets:
        results = []
        logger.info('Testing on {} dataset'.format(name))
        num_proposals = 0
        total_time = 0

        from collections import defaultdict
        time_dict = defaultdict(float)
        
        data_list = []
        for i, (images, annotations) in enumerate(dataset):
            data_list.append((images.to(device),annotations))
        for i, (images, annotations) in enumerate(tqdm(data_list)):
            with torch.no_grad():
                output, extra_info = model(images, annotations=annotations)
            output = to_device(output,'cpu')
            num_proposals += output['num_proposals']/len(dataset)
            for key, val in extra_info.items():
                time_dict[key]+=val
        
            for k in output.keys():
                if isinstance(output[k], torch.Tensor):
                    output[k] = output[k].tolist()
            results.append(output)

        total_time = sum(time_dict.values())
        logger.info('FPS = {}'.format(len(dataset)/total_time))
        logger.info('Number of proposals: {}'.format(num_proposals))
        outpath_dataset = osp.join(root,'{}.json'.format(name))

        logger.info('Writing the results of the {} dataset into {}'.format(name,
                    outpath_dataset))
        
        with open(outpath_dataset,'w') as _out:
            json.dump(results,_out)

        logger.info('evaluating the results on the {} dataset'.format(name))
        ann_file = DatasetCatalog.get(name)['args']['ann_file']
        with open(ann_file,'r') as _ann:
            annotations_list = json.load(_ann)
        annotations_dict = {
            ann['filename']: ann for ann in annotations_list
        }
        with open(outpath_dataset,'r') as _res:
            result_list = json.load(_res)
        
        for threshold in THRESHOLDS:
            sAP, P, R, F = sAPEval(result_list, annotations_dict, threshold)
            sAP_string = 'sAP{} = {:.1f}'.format(threshold,sAP*100)   
            logger.info(sAP_string)
            logger.info('sF-{} = {:.1f}'.format(threshold,F*100))
