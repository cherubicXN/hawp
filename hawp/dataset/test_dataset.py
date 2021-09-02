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

def collate_images_anns_meta(batch):
    anns = [b[-2] for b in batch]
    metas = [b[-1] for b in batch]
    if len(batch[0]) == 4:
        images = [b[0] for b in batch]
        processed_images = default_collate([b[1] for b in batch])
        return images, processed_images, anns, metas
    
    processed_images = default_collate([b[0] for b in batch])
    return processed_images, anns, metas


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
        image = io.imread(osp.join(self.root,ann['filename'])).astype(float)[:,:,:3]
        for key, _type in (['junc',np.float32],
                            ['lines',  np.float32]):
            ann[key] = np.array(ann[key],dtype=_type)

        meta = {
            'filename': osp.join(self.root,ann['filename']),
            'height': image.shape[0],
            'width': image.shape[1],
        }
        if self.transform is not None:
            return self.transform(image,ann)
        return image, ann, meta
    def image(self, idx):
        ann = copy.deepcopy(self.annotations[idx])
        image = Image.open(osp.join(self.root,ann['filename'])).convert('RGB')
        return image

class ImageList(Dataset):
    def __init__(self, image_paths, transform=None, with_raw_image=False):
        super().__init__()
        self.image_paths = image_paths
        self.transform = transform
        self.with_raw_image = with_raw_image
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = io.imread(image_path)
        anns = {}
        meta = {
            'dataset_index': index,
            'filename': image_path,
            'height': image.shape[0],
            'width': image.shape[1],
        }
        processed_image, anns = self.transform(image.astype(float)[:,:,:3],anns)
        if self.with_raw_image:
            return image, processed_image, anns, meta
        return processed_image, anns, meta


class NumpyImageList(Dataset):
    def __init__(self, images, transform=None, with_raw_image=False):
        super().__init__()
        self.images = images
        self.transform = transform
        self.with_raw_image = with_raw_image
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        image = Image.fromarray(self.images[index])
        anns = {}
        meta = {
            'dataset_index': index,
            'filename': '{:09d}'.format(index),
            'height': image.shape[0],
            'width': image.shape[1],
        }
        processed_image, anns = self.transform(image.astype(float)[:,:,:3],anns)
        if self.with_raw_image:
            return image, processed_image, anns, meta
        return processed_image, anns, meta

        