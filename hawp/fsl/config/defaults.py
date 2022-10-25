from yacs.config import CfgNode as CN
from .models import MODELS
from .dataset import DATASETS
from .solver import SOLVER
from .detr import DETR
cfg = CN()

cfg.ENCODER = CN()
cfg.ENCODER.DIS_TH = 5
cfg.ENCODER.ANG_TH = 0.1
cfg.ENCODER.NUM_STATIC_POS_LINES = 300
cfg.ENCODER.NUM_STATIC_NEG_LINES = 40
cfg.ENCODER.BACKGROUND_WEIGHT = 0.0
cfg.MODELING_PATH = 'hawp'
cfg.MODEL = MODELS
cfg.DATASETS = DATASETS
cfg.SOLVER = SOLVER

cfg.DATALOADER = CN()
cfg.DATALOADER.NUM_WORKERS = 8
cfg.OUTPUT_DIR = "outputs/dev"
