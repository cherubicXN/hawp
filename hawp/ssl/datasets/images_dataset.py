import os
import os.path as osp
import glob
import math
import copy
from skimage.io import imread
from skimage import color
import PIL
import numpy as np
import h5py
import cv2
import pickle
import torch
import torch.utils.data.dataloader as torch_loader
from torch.utils.data import Dataset
from torchvision import transforms

from ..config.project_config import Config as cfg
from .transforms import photometric_transforms as photoaug
from .transforms import homographic_transforms as homoaug
from .transforms.utils import random_scaling
from .synthetic_util import get_line_heatmap
from ..misc.train_utils import parse_h5_data
from ..misc.geometry_utils import warp_points, mask_points
from tqdm import tqdm

def images_collate_fn(batch):
    """ Customized collate_fn for wireframe dataset. """
    batch_keys = ["image", "junction_map", "valid_mask", "heatmap",
                  "heatmap_pos", "heatmap_neg", "homography",
                  "line_points", "line_indices"]
    list_keys = ["junctions", "line_map", "line_map_pos",
                 "line_map_neg", "file_key","fname"]

    outputs = {}
    for data_key in batch[0].keys():
        batch_match = sum([_ in data_key for _ in batch_keys])
        list_match = sum([_ in data_key for _ in list_keys])
        # print(batch_match, list_match)
        if batch_match > 0 and list_match == 0:
            outputs[data_key] = torch_loader.default_collate(
                [b[data_key] for b in batch])
        elif batch_match == 0 and list_match > 0:
            outputs[data_key] = [b[data_key] for b in batch]
        elif batch_match == 0 and list_match == 0:
            continue
        else:
            raise ValueError(
        "[Error] A key matches batch keys and list keys simultaneously.")

    return outputs

class ImageCollections(Dataset):
    def __init__(self, mode, config):
        super(ImageCollections, self).__init__()
        if config is None:
            self.config = self.get_default_config()
        else:
            self.config = config
        self.filenames = sorted(glob.glob(osp.join(self.config['dataset_root'],self.config['img_reg_exp'])))
        self.default_config = self.get_default_config()

        self.dataset_name = self.config['alias']

        self.size = self.config['preprocessing']['resize']

        self.h5path = self.config.get('labels',None) 
        if self.config.get('load_labels',False) is False:
            self.h5path = None
        
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        fname = osp.basename(self.filenames[idx])
        image = cv2.imread(self.filenames[idx], int(self.config['gray_scale']==False))
        image = cv2.resize(image, self.size)
        image = np.array(image,dtype=np.float32)/255.0

        data = {
            'fname': fname,
            'image': torch.from_numpy(image)[None],
            'valid_mask': torch.ones(self.size,dtype=torch.float32)[None]
        }

        if self.h5path is not None:
            with h5py.File(self.h5path,'r') as f:
                exported_label = parse_h5_data(f[fname])
            assert all(exported_label['size'] == self.size)

            junctions_xy = exported_label["junctions"]
            junctions = junctions_xy[:,[1,0]]
            line_map = exported_label["line_map"]
            heatmap = get_line_heatmap(junctions_xy,line_map,self.size)[None]
            data['line_map'] = torch.tensor(line_map).float()
            data['junctions'] = torch.tensor(junctions).float()
            data['heatmap'] = torch.tensor(heatmap).float()

        return data
        
    def get_default_config(self):
        return {
            "dataset_name": "images",
            "add_augmentation_to_all_splits": False,
            "preprocessing": {
                "resize": [512,512],
                "blur_size": 11,
            },
            "augmentation": {
                "photometric": {
                    "enable": False
                },
                "homographic": {
                    "enable": False
                }
            }
        }