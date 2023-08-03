import torch
import random 
import numpy as np
import os
import os.path as osp
from tqdm import tqdm

from hawp.base import to_device, setup_logger, MetricLogger, save_config
from hawp.base import show

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
    show.Canvas.show = True
    painter = show.painters.HAWPainter()

    for fname in tqdm(args.img):
        image = cv2.imread(fname,0)
        
        # import pdb; pdb.set_trace()
        ori_shape = image.shape[:2]
        image_cp = copy.deepcopy(image)
        image_ = cv2.resize(image_cp,(width,height))
        image_ = torch.from_numpy(image_).float()/255.0
        image_ = image_[None,None].cuda()
        
        
        meta = {
            'width': ori_shape[1],
            'height':ori_shape[0],
            'filename': ''
        }
        with torch.no_grad():
            outputs, _ = model(image_,[meta])


        with show.image_canvas(fname) as ax:
            painter.draw_wireframe(ax,outputs)


if __name__ == "__main__":
    main()
