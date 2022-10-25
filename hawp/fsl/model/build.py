from . import models


def build_model(cfg):
    model = models.WireframeDetector(cfg)

    return model