from .registry import MODELS
from .stacked_hg import HourglassNet, Bottleneck2D
from .multi_task_head import MultitaskHead
from .resnets import ResNets

@MODELS.register("Hourglass")
def build_hg(cfg, **kwargs):
    inplanes = cfg.MODEL.HGNETS.INPLANES
    num_feats = cfg.MODEL.OUT_FEATURE_CHANNELS//2
    depth = cfg.MODEL.HGNETS.DEPTH
    num_stacks = cfg.MODEL.HGNETS.NUM_STACKS
    num_blocks = cfg.MODEL.HGNETS.NUM_BLOCKS
    head_size = cfg.MODEL.HEAD_SIZE

    out_feature_channels = cfg.MODEL.OUT_FEATURE_CHANNELS

    if kwargs.get('gray_scale',False):
        input_channels = 1
    else:
        input_channels = 3
    num_class = sum(sum(head_size, []))
    model = HourglassNet(
        input_channels=input_channels,
        block=Bottleneck2D,
        inplanes = inplanes,
        num_feats= num_feats,
        depth=depth,
        head=lambda c_in, c_out: MultitaskHead(c_in, c_out, head_size=head_size),
        num_stacks = num_stacks,
        num_blocks = num_blocks,
        num_classes = num_class)

    model.out_feature_channels = out_feature_channels

    return model


# @MODELS.register("ResNets")
# def build_resnet(cfg):
#     head_size = cfg.MODEL.HEAD_SIZE

#     num_class = sum(sum(head_size,[]))
#     model = ResNets(cfg.MODEL.RESNETS.BASENET,head=lambda c_in, c_out: MultitaskHead(c_in, c_out, head_size=head_size),num_class=num_class,pretrain=cfg.MODEL.RESNETS.PRETRAIN)

#     model.out_feature_channels = 128
#     return model
    
def build_backbone(cfg, **kwargs):
    assert cfg.MODEL.NAME in MODELS,  \
        "cfg.MODELS.NAME: {} is not registered in registry".format(cfg.MODELS.NAME)

    return MODELS[cfg.MODEL.NAME](cfg, **kwargs)
