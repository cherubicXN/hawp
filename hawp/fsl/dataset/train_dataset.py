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
    def __init__(self, root, ann_file, transform = None, augmentation = 4):
        self.root = root
        with open(ann_file,'r') as _:
            self.annotations = json.load(_)
        self.transform = transform
        self.augmentation = augmentation
    
    def __getitem__(self, idx_):
        # print(idx_)

        idx = idx_%len(self.annotations)
        # random_prob = torch.rand(1)
        # reminder = torch.randint(0,4,(1,)).item()
        reminder = idx_//len(self.annotations)
        
        # idx = 0
        # reminder = 0
        ann = copy.deepcopy(self.annotations[idx])
        if len(ann['edges_negative']) == 0:
            ann['edges_negative'] = [[0,0]]
        ann['reminder'] = reminder
        image = io.imread(osp.join(self.root,ann['filename'])).astype(float)#[:,:,:3]
        if len(image.shape) == 2:
            image = np.concatenate([image[...,None],image[...,None],image[...,None]],axis=-1)
        else:
            image = image[:,:,:3]

        # if len(ann['junctions']) == 0:
        #     ann['junctions'] = [[0,0]]
        #     ann['edges_positive'] = [[0,0]]
        
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
        elif reminder == 4:
            image_rotated = np.rot90(image)

            junctions = ann['junctions'] - np.array([image.shape[1],image.shape[0]]).reshape(1,-1)/2.0
            theta = 0.5*np.pi
            rot_mat = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
            junctions_rotated = (rot_mat@junctions.transpose()).transpose()
            junctions_rotated = junctions_rotated + np.array([image_rotated.shape[1],image_rotated.shape[0]]).reshape(1,-1)/2.0

            ann['width'] = image_rotated.shape[1]
            ann['height'] = image_rotated.shape[0]
            ann['junctions'] = np.asarray(junctions_rotated,dtype=np.float32)
            image = image_rotated
        elif reminder == 5:
        #     image_rotated = np.rot90(np.rot90(image))

        #     junctions = ann['junctions'] - np.array([image.shape[1],image.shape[0]]).reshape(1,-1)/2.0
        #     theta = np.pi
        #     rot_mat = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
        #     junctions_rotated = (rot_mat@junctions.transpose()).transpose()
        #     junctions_rotated = junctions_rotated + np.array([image_rotated.shape[1],image_rotated.shape[0]]).reshape(1,-1)/2.0

        #     ann['width'] = image_rotated.shape[1]
        #     ann['height'] = image_rotated.shape[0]
        #     ann['junctions'] = np.asarray(junctions_rotated,dtype=np.float32)
        #     image = image_rotated
        # elif reminder == 6:
            image_rotated = np.rot90(np.rot90(np.rot90(image)))

            junctions = ann['junctions'] - np.array([image.shape[1],image.shape[0]]).reshape(1,-1)/2.0
            theta = 1.5*np.pi
            rot_mat = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
            junctions_rotated = (rot_mat@junctions.transpose()).transpose()
            junctions_rotated = junctions_rotated + np.array([image_rotated.shape[1],image_rotated.shape[0]]).reshape(1,-1)/2.0

            ann['width'] = image_rotated.shape[1]
            ann['height'] = image_rotated.shape[0]
            ann['junctions'] = np.asarray(junctions_rotated,dtype=np.float32)
            image = image_rotated
        if self.transform is not None:
            return self.transform(image,ann)
        
        
        return image, ann

    def __len__(self):
        return len(self.annotations)*self.augmentation

def collate_fn(batch):
    return (default_collate([b[0] for b in batch]),
            [b[1] for b in batch])
