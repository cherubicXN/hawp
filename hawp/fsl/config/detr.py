from yacs.config import CfgNode as CN


DETR = CN()

DETR.backbone = 'resnet50'
DETR.dilation = False #dilated conv, DC5 for DETR
DETR.position_embedding = 'sine'
DETR.lr_backbone = 1e-5
DETR.enc_layers = 6
DETR.dec_layers = 6
DETR.dim_feedforward = 2048
DETR.hidden_dim = 256
DETR.dropout = 0.1
DETR.nheads = 8
DETR.num_queries = 1000
DETR.pre_norm = False
DETR.eos_coef = 0.1 #"Relative classification weight of the no-object class"

DETR.no_aux_loss = False
DETR.set_cost_class = 1.0
DETR.set_cost_lines = 5.0


