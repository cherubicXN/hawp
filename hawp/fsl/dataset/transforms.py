import torch
import torchvision
from torchvision.transforms import functional as F
from skimage.transform  import resize
import cv2
import numpy as np
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, ann=None):
        if ann is None:
            for t in self.transforms:
                image = t(image)
            return image
        for t in self.transforms:
            image, ann = t(image, ann)
        return image, ann
class Resize(object):
    def __init__(self, image_height, image_width, ann_height, ann_width):
        self.image_height = image_height
        self.image_width  = image_width
        self.ann_height   = ann_height
        self.ann_width    = ann_width

    def __call__(self, image, ann):
        image = resize(image,(self.image_height,self.image_width))
        image = np.array(image,dtype=np.float32)/255.0

        sx = self.ann_width/ann['width']
        sy = self.ann_height/ann['height']
        ann['junctions'][:,0] = np.clip(ann['junctions'][:,0]*sx, 0, self.ann_width-1e-4)
        ann['junctions'][:,1] = np.clip(ann['junctions'][:,1]*sy, 0, self.ann_height-1e-4)
        ann['width'] = self.ann_width
        ann['height'] = self.ann_height
        
        return image, ann

class RandomRotation(object):
    def __init__(self, image_height, image_width, target_height, target_width):
        self.image_height = image_height
        self.image_width = image_width
        self.target_height = target_height
        self.target_width = target_width
        
    def __call__(self, image, ann):
        prob = torch.rand(1)
        if prob<=0.25:
            rotation = 0
        elif prob<=0.5:
            rotation = 90
        elif prob<=0.75:
            rotation = 180
        else:
            rotation = 270
        
        center = (self.image_width/2.0, self.image_height/2.0)
        center_tgt = (self.target_width/2.0, self.target_height/2.0)
        mat_im = cv2.getRotationMatrix2D(center, rotation, 1.0)
        mat_tgt = cv2.getRotationMatrix2D(center_tgt, rotation, 1.0)
        image_rotated = cv2.warpAffine(image, mat_im, (self.image_width, self.image_height))

        junctions = ann['junctions']
        X = np.concatenate((junctions,np.ones((junctions.shape[0],1))),axis=-1,dtype=np.float32)
        X_rotated = (mat_tgt@(X.transpose())).transpose()
        X_rotated = np.asarray(X_rotated,dtype=np.float32)
        X_rotated[:,0] = np.clip(X_rotated[:,0],0,self.target_width-1e-4)
        X_rotated[:,1] = np.clip(X_rotated[:,1],0,self.target_height-1e-4)
        # edges = ann['edges_positive']
        # lines = X_rotated[edges].reshape(-1,4)
        # import matplotlib.pyplot as plt
        # plt.imshow(image_rotated)
        # plt.plot([lines[:,0]*4,lines[:,2]*4],[lines[:,1]*4,lines[:,3]*4],'r-')
        # plt.show()
        
        ann['junctions'] = X_rotated

        return image_rotated, ann
        # return image, ann
        
class ResizeImage(object):
    def __init__(self, image_height, image_width):
        self.image_height = image_height
        self.image_width  = image_width

    def __call__(self, image, ann=None):
        # image = resize(image,(self.image_height,self.image_width))
        image = cv2.resize(image,(self.image_width,self.image_height))
        image = np.array(image,dtype=np.float32)/255.0
        if ann is None:
            return image
        return image, ann

class ToTensor(object):
    def __call__(self, image, anns=None):
        if anns is None:
            return F.to_tensor(image)
            
        for key,val in anns.items():
            if isinstance(val,np.ndarray):
                anns[key] = torch.from_numpy(val)
        return F.to_tensor(image),anns

class Normalize(object):
    def __init__(self, mean, std,to_255 = False):
        self.mean = mean
        self.std  = std
        self.to_255 = to_255
    def __call__(self, image, anns = None):
        if self.to_255:
            image*=255.0
        image = F.normalize(image,mean=self.mean,std=self.std)
        if anns is None:
            return image
        return image, anns
        