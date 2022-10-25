import argparse
import logging
import time

import numpy as np
import torch
import torchvision
try:
    import cv2  # pylint: disable=import-error
except ImportError:
    cv2 = None

import PIL
try:
    import PIL.ImageGrab
except ImportError:
    pass

try:
    import mss
except ImportError:
    mss = None

import os
import os.path as osp
LOG = logging.getLogger(__name__)

class ToTensor(object):
    def __call__(self,image):
        tensor = torch.from_numpy(image).float()/255.0
        tensor = tensor.permute((2,0,1)).contiguous()
        return tensor

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, image):
        image[0] = (image[0]-self.mean[0])/self.std[0]
        image[1] = (image[1]-self.mean[1])/self.std[1]
        image[2] = (image[2]-self.mean[2])/self.std[2]
        return image

# pylint: disable=abstract-method
class ImageList(torch.utils.data.Dataset):
    horizontal_flip = None
    rotate = None
    crop = None
    scale = 1.0
    start_frame = None
    start_msec = None
    max_frames = None

    def __init__(self, source, *,
                 input_size,
                 transforms,
                 ):
        super().__init__()

        self.source = source
        self.input_size = input_size
        self.transforms = transforms
        
        self.filenames = sorted(os.listdir(source))

    def __len__(self):
        return len(self.filenames)
    
    # pylint: disable=unsubscriptable-object
    def preprocessing(self, image):
        if self.scale != 1.0:
            image = cv2.resize(image, None, fx=self.scale, fy=self.scale)
            LOG.debug('resized image size: %s', image.shape)
        if self.horizontal_flip:
            image = image[:, ::-1]
        if self.crop:
            if self.crop[0]:
                image = image[:, self.crop[0]:]
            if self.crop[1]:
                image = image[self.crop[1]:, :]
            if self.crop[2]:
                image = image[:, :-self.crop[2]]
            if self.crop[3]:
                image = image[:-self.crop[3], :]
        if self.rotate == 'left':
            image = np.swapaxes(image, 0, 1)
            image = np.flip(image, axis=0)
        elif self.rotate == 'right':
            image = np.swapaxes(image, 0, 1)
            image = np.flip(image, axis=1)
        elif self.rotate == '180':
            image = np.flip(image, axis=0)
            image = np.flip(image, axis=1)

        meta = {
            'width': image.shape[1],
            'height': image.shape[0],
        }

        processed_image = self.transforms(image)
        
        return image, processed_image, meta

    def __getitem__(self, id):
        fname = osp.join(self.source,self.filenames[id])

        image = cv2.imread(fname)
        meta = {
            'width': image.shape[1],
            'height': image.shape[0],
        }
        meta['frame_i'] = id
        meta['filename'] = '{:05d}'.format(id)
        processed_image = self.transforms(image)
        return image, processed_image, meta
        