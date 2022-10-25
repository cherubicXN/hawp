import torch
from .transforms import *
from . import train_dataset
from ..config.paths_catalog import DatasetCatalog
from . import test_dataset

def build_transform(cfg):
    transforms = Compose(
        [ResizeImage(cfg.DATASETS.IMAGE.HEIGHT,
                     cfg.DATASETS.IMAGE.WIDTH),
         ToTensor(),
         Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                                           cfg.DATASETS.IMAGE.PIXEL_STD,
                                           cfg.DATASETS.IMAGE.TO_255)
        ]
    )
    return transforms
def build_train_dataset(cfg):
    assert len(cfg.DATASETS.TRAIN) == 1
    name = cfg.DATASETS.TRAIN[0]
    dargs = DatasetCatalog.get(name)

    factory = getattr(train_dataset,dargs['factory'])
    args = dargs['args']
    args['augmentation'] = cfg.DATASETS.AUGMENTATION
    args['transform'] = Compose(
                                [Resize(cfg.DATASETS.IMAGE.HEIGHT,
                                        cfg.DATASETS.IMAGE.WIDTH,
                                        cfg.DATASETS.TARGET.HEIGHT,
                                        cfg.DATASETS.TARGET.WIDTH),
                                 ToTensor(),
                                 Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                                           cfg.DATASETS.IMAGE.PIXEL_STD,
                                           cfg.DATASETS.IMAGE.TO_255)])
    dataset = factory(**args)
    
    dataset = torch.utils.data.DataLoader(dataset,
                                          batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                          collate_fn=train_dataset.collate_fn,
                                          shuffle = True,
                                          num_workers = cfg.DATALOADER.NUM_WORKERS)
    return dataset

def build_test_dataset(cfg):
    transforms = Compose(
        [ResizeImage(cfg.DATASETS.IMAGE.HEIGHT,
                     cfg.DATASETS.IMAGE.WIDTH),
         ToTensor(),
         Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                                           cfg.DATASETS.IMAGE.PIXEL_STD,
                                           cfg.DATASETS.IMAGE.TO_255)
        ]
    )

    datasets = []
    for name in cfg.DATASETS.TEST:
        dargs = DatasetCatalog.get(name)
        factory = getattr(test_dataset,dargs['factory'])
        args = dargs['args']
        args['transform'] = transforms
        dataset = factory(**args)
        dataset = torch.utils.data.DataLoader(
            dataset,  batch_size = 1,
            collate_fn = dataset.collate_fn,
            num_workers = cfg.DATALOADER.NUM_WORKERS,
        )
        datasets.append((name,dataset))
    return datasets
