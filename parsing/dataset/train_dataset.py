import torch
from torch.utils.data import Dataset

import os.path as osp
import json
import cv2
from skimage import io
from PIL import Image
import numpy as np
import random
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
import copy
class TrainDataset(Dataset):
    def __init__(self, root, ann_file, transform = None):
        self.root = root
        with open(ann_file,'r') as _:
            self.annotations = json.load(_)
        self.transform = transform
    
    def __getitem__(self, idx_):
        # print(idx_)
        idx = idx_%len(self.annotations)
        reminder = idx_//len(self.annotations)
        ann = copy.deepcopy(self.annotations[idx])
        ann['reminder'] = reminder
        image = io.imread(osp.join(self.root,ann['filename'])).astype(float)[:,:,:3]
        # image = Image.open(osp.join(self.root,ann['filename'])).convert('RGB')
        for key,_type in (['junctions',np.float32],
                          ['edges_positive',np.long],
                          ['edges_negative',np.long]):
            ann[key] = np.array(ann[key],dtype=_type)
        
        width = ann['width']
        height = ann['height']
        if reminder == 1:
            image = image[:,::-1,:]
            # image = F.hflip(image)
            ann['junctions'][:,0] = width-ann['junctions'][:,0]
        elif reminder == 2:
            # image = F.vflip(image)
            image = image[::-1,:,:]
            ann['junctions'][:,1] = height-ann['junctions'][:,1]
        elif reminder == 3:
            # image = F.vflip(F.hflip(image))
            image = image[::-1,::-1,:]
            ann['junctions'][:,0] = width-ann['junctions'][:,0]
            ann['junctions'][:,1] = height-ann['junctions'][:,1]
        else:
            pass

        if self.transform is not None:
            return self.transform(image,ann)
        return image, ann

    def __len__(self):
        return len(self.annotations)*4

def collate_fn(batch):
    return (default_collate([b[0] for b in batch]),
            [b[1] for b in batch])