import os
import os.path as osp

class DatasetCatalog(object):

    DATA_DIR = osp.abspath(osp.join(osp.dirname(__file__),
                '..','..','data'))
    
    DATASETS = {
        'wireframe_train': {
            'img_dir': 'wireframe/images',
            'ann_file': 'wireframe/train.json',
        },
        'wireframe_test': {
            'img_dir': 'wireframe/images',
            'ann_file': 'wireframe/test.json',
        },
        'york_test': {
            'img_dir': 'york/images',
            'ann_file': 'york/test.json',
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