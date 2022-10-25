import os
import os.path as osp

class DatasetCatalog(object):

    DATA_DIR = osp.abspath(osp.join(osp.dirname(__file__),
                '..','..','..','data'))
    
    DATASETS = {
        'wireframe_train': {
            'img_dir': 'wireframe/images',
            'ann_file': 'wireframe/train.json',
        },
        'wireframe_train-pseudo': {
            'img_dir': 'wireframe-pseudo/images',
            'ann_file': 'wireframe-pseudo/train.json',
        },
        'wireframe_train-syn-export': {
            'img_dir': 'wireframe-syn-export/images',
            'ann_file': 'wireframe-syn-export/train.json',
        },
        'wireframe_train-syn-export-1': {
            'img_dir': 'wireframe-syn-export-ep30-iter100-th075/images',
            'ann_file': 'wireframe-syn-export-ep30-iter100-th075/train.json',
        },
        'wireframe_test1': {
            'img_dir': 'wireframe/images',
            'ann_file': 'wireframe/overfit.json',
        },
        'synthetic_train': {
            'img_dir': 'synthetic-shapes/images',
            'ann_file': 'synthetic-shapes/train.json',
        },
        'synthetic_test': {
            'img_dir': 'synthetic-shapes/images',
            'ann_file': 'synthetic-shapes/test.json',
        },
        'cities_train': {
            'img_dir': 'cities/images',
            'ann_file': 'cities/train.json',
        },
        'cities_test': {
            'img_dir': 'cities/images',
            'ann_file': 'cities/test.json',
        },
        'wireframe_test': {
            'img_dir': 'wireframe/images',
            'ann_file': 'wireframe/test.json',
        },
        'york_test': {
            'img_dir': 'york/images',
            'ann_file': 'york/test.json',
        },
        'coco_train-val2017': {
            'img_dir': 'coco/val2017',
            'ann_file': 'coco/coco-wf-val.json',
        },
        'coco_test-val2017': {
            'img_dir': 'coco/val2017',
            'ann_file': 'coco/coco-wf-val.json',
        }
    }

    @staticmethod
    def get(name):
        assert name in DatasetCatalog.DATASETS
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]

        args = dict(
            root = osp.join(data_dir,attrs['img_dir']),
            ann_file = osp.join(data_dir,attrs['ann_file'])
        )

        if 'train' in name:
            return dict(factory="TrainDataset",args=args)
        if 'test' in name and 'ann_file' in attrs:
            return dict(factory="TestDatasetWithAnnotations",
                        args=args)
        raise NotImplementedError()