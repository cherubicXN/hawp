import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import json
import copy
from PIL import Image
from skimage import io
import os
import os.path as osp
import numpy as np
class TestDatasetWithAnnotations(Dataset):
    '''
    Format of the annotation file
    annotations[i] has the following dict items:
    - filename  # of the input image, str 
    - height    # of the input image, int
    - width     # of the input image, int
    - lines     # of the input image, list of list, N*4
    - junc      # of the input image, list of list, M*2
    '''

    def __init__(self, root, ann_file, transform = None):
        self.root = root
        with open(ann_file, 'r') as _:
            self.annotations = json.load(_)
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = copy.deepcopy(self.annotations[idx])
        # image = Image.open(osp.join(self.root,ann['filename'])).convert('RGB')
        image = io.imread(osp.join(self.root,ann['filename'])).astype(float)[:,:,:3]
        for key, _type in (['junc',np.float32],
                            ['lines',  np.float32]):
            ann[key] = np.array(ann[key],dtype=_type)

        if self.transform is not None:
            return self.transform(image,ann)
        return image, ann
    def image(self, idx):
        ann = copy.deepcopy(self.annotations[idx])
        image = Image.open(osp.join(self.root,ann['filename'])).convert('RGB')
        return image
    @staticmethod
    def collate_fn(batch):
        return (default_collate([b[0] for b in batch]),
                [b[1] for b in batch])
    