import os
import sys
import glob
import os.path as osp

import cv2
import numpy as np
import scipy.io as sio
import matplotlib as mpl
import numpy.linalg as LA
import matplotlib.pyplot as plt
import argparse
import json
from tqdm import tqdm
import time
DATASET_ROOT = 'epnet/data/%s'
if __name__ == '__main__':

    parser = argparse.ArgumentParser('program to run LSD on datasets')
    parser.add_argument('--dataset',type=str,required=True)
    # parser.add_argument('--mode',type=str,default='fast')


    args = parser.parse_args()

    assert args.dataset == 'wireframe' or 'york'
    # assert args.mode    == 'fast' or 'normal'

    dataset_path = DATASET_ROOT%args.dataset
    img_dir = osp.join(dataset_path,'images/%s')
    with open(osp.join(dataset_path,'test.json'),'r') as buffer:
        annotations = json.load(buffer)

    image_list = []

    print('Loading images to memory')
    for anno in tqdm(annotations):
        image = cv2.imread(img_dir%anno['filename'],0)
        image_list.append(image)


    outdir = 'lsd/%s'%args.dataset
    os.makedirs(outdir,exist_ok=True)

    start = time.time()
    for img, anno in zip(tqdm(image_list),annotations):
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
        lines, _, _, score = lsd.detect(img)
        lines = lines.squeeze()
        idx   = np.argsort(-score.flatten())
        lines = np.concatenate((lines,score),axis=1)[idx]
        lines = np.array(lines,dtype=np.float32)
        mdict = {
            'height': anno['height'],
            'width' : anno['width'],
            'gt'    : np.array(anno['lines'],dtype=np.float32),
            'pred'  : lines
        }
        fname = anno['filename'].replace('.png','.mat')
        sio.savemat(osp.join(outdir,fname),mdict=mdict)


    end = time.time()


    FPS = (end-start)/float(len(annotations))
    # print((end-start)/)

    # img = cv2.imread('epnet/data/wireframe/images/00064981.png',0)
    # out = lsd.detect(img)
