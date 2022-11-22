import torch
import random 
import numpy as np
import os
import os.path as osp
from tqdm import tqdm
#import hawp
from hawp.base.utils.comm import to_device
from hawp.base.utils.logger import setup_logger
from hawp.base.utils.metric_logger import MetricLogger
from hawp.base.utils.miscellaneous import save_config

from hawp.fsl.solver import make_lr_scheduler, make_optimizer
from hawp.fsl.config import cfg as model_config

from hawp.ssl.config import Config, load_config
from hawp.ssl.datasets import dataset_util
from hawp.ssl.models import MODELS


from torch.utils.data import DataLoader
import torch.utils.data.dataloader as torch_loader

from pathlib import Path
import argparse
import yaml
import logging
import time
import datetime
import cv2
import copy
import sys
def parse_args():
    aparser = argparse.ArgumentParser()
    aparser.add_argument('--metarch', default='HAWP', choices=MODELS.keys())
    aparser.add_argument('--cfg', default=os.path.join(
        os.path.dirname(__file__),'config','hawpv3.yaml'), 
        help='model configuration')
    aparser.add_argument('--ckpt', required=True, help='checkpoint')
    aparser.add_argument('-t','--threshold', default=0.5,type=float)
    aparser.add_argument('--img', required=True,type=str, nargs='+')
    aparser.add_argument('--width', default=512,type=int)
    aparser.add_argument('--height', default=512,type=int)
    aparser.add_argument('--whitebg', default=0.0, type=float)
    aparser.add_argument('--saveto', default=None, type=str,)
    aparser.add_argument('--ext', default='pdf', type=str, choices=['pdf','png'])

    args = aparser.parse_args()

    return args

def main():
    args = parse_args()
    model_config.merge_from_file(args.cfg)
    
    model = MODELS[args.metarch](model_config, gray_scale=True)
    model = model.eval().cuda()

    weight_path = args.ckpt
    state_dict = torch.load(weight_path)

    model.load_state_dict(state_dict)
    

    imagepath_list = [args.img]

    height = args.height
    width = args.width

    if args.saveto:
        os.makedirs(args.saveto,exist_ok=True)
        DEST = args.saveto
        
    for fname in tqdm(args.img):
        image = cv2.imread(fname,0)
        ori_shape = image.shape[:2]
        image_cp = copy.deepcopy(image)
        image_ = cv2.resize(image_cp,(width,height))
        image_ = torch.from_numpy(image_).float()/255.0
        image_ = image_[None,None].cuda()
        
        
        meta = {
            'width': image_.shape[-1],
            'height':image_.shape[-2],
            'filename': ''
        }
        with torch.no_grad():
            outputs, _ = model(image_,[meta])

        sx = ori_shape[1]/float(width)
        sy = ori_shape[0]/float(height)

        lines = outputs['lines_pred'].cpu().numpy()
        lines *= np.array([sx,sy,sx,sy]).reshape(-1,4)
        scores_pred = outputs['lines_score'].cpu().numpy()
        import matplotlib.pyplot as plt
        # plt.imshow(image[0,0].cpu())
        fig = plt.figure()
        fig.set_size_inches(ori_shape[1]/ori_shape[0],1,forward=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.xlim([-0.5, ori_shape[1]-0.5])
        plt.ylim([ori_shape[0]-0.5, -0.5])
        # if args.whitebg:
            # plt.imshow(cv2.imread(fname)[:,:,::-1]*0+255)
        # else:
        plt.imshow(cv2.imread(fname)[:,:,::-1],alpha=1-args.whitebg)
        th = args.threshold
        plt.plot([lines[scores_pred>th,0],lines[scores_pred>th,2]],[lines[scores_pred>th,1],lines[scores_pred>th,3]],'r-', linewidth = 0.5 if args.saveto else None)
        plt.scatter(lines[scores_pred>th,0],lines[scores_pred>th,1],color='b',s=1.2,edgecolors='none',zorder=5)
        plt.scatter(lines[scores_pred>th,2],lines[scores_pred>th,3],color='b',s=1.2,edgecolors='none',zorder=5)
        plt.axis('off')

        if args.saveto:
            print(osp.join(DEST,osp.basename(fname[:-4])+'.'+args.ext),(scores_pred>th).sum())
            plt.savefig(osp.join(DEST,osp.basename(fname[:-4])+'.'+args.ext),dpi=300,bbox_inches=0)
            plt.close()
        else:
            plt.show()

if __name__ == "__main__":
    main()
