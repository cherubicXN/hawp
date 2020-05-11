from .registry import MODELS
from .stacked_hg import HourglassNet, Bottleneck2D
from .multi_task_head import MultitaskHead

@MODELS.register("Hourglass")
def build_hg(cfg):
    inplanes = cfg.MODEL.HGNETS.INPLANES
    num_feats = cfg.MODEL.OUT_FEATURE_CHANNELS//2
    depth = cfg.MODEL.HGNETS.DEPTH
    num_stacks = cfg.MODEL.HGNETS.NUM_STACKS
    num_blocks = cfg.MODEL.HGNETS.NUM_BLOCKS
    head_size = cfg.MODEL.HEAD_SIZE

    out_feature_channels = cfg.MODEL.OUT_FEATURE_CHANNELS


    num_class = sum(sum(head_size, []))
    model = HourglassNet(
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


def build_backbone(cfg):
    assert cfg.MODEL.NAME in MODELS,  \
        "cfg.MODELS.NAME: {} is not registered in registry".format(cfg.MODELS.NAME)

    return MODELS[cfg.MODEL.NAME](cfg)
