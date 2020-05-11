from yacs.config import CfgNode as CN
from .shg import HGNETS
from .head import PARSING_HEAD

MODELS = CN()

MODELS.NAME = "Hourglass"
MODELS.HGNETS = HGNETS
MODELS.DEVICE = "cuda"
MODELS.WEIGHTS = ""
MODELS.HEAD_SIZE  = [[3], [1], [1], [2], [2]] 
MODELS.OUT_FEATURE_CHANNELS = 256

MODELS.LOSS_WEIGHTS = CN(new_allowed=True)

MODELS.PARSING_HEAD   = PARSING_HEAD
MODELS.SCALE = 1.0