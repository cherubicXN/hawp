import os
import torch
import argparse
from skimage import io
import matplotlib.pyplot as plt

from parsing.config import cfg
from parsing.utils.comm import to_device
from parsing.utils.logger import setup_logger
from parsing.detector_onnx import get_hawp_model
from parsing.dataset.build import build_transform
from parsing.utils.checkpoint import DetectronCheckpointer


parser = argparse.ArgumentParser()

parser.add_argument(
    "--config-file",
    metavar="FILE",
    type=str,
    default=None,
)
parser.add_argument("--img", type=str, required=True)

parser.add_argument("--line_threshold", type=float, default=0.5)

parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()


def onnx(cfg):
    impath = args.img
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    model = get_hawp_model(pretrained=args.config_file is None)
    model = model.to(device)

    transform = build_transform(cfg)
    output_dir = cfg.OUTPUT_DIR

    if args.config_file is not None:
        checkpointer = DetectronCheckpointer(
            cfg, model, save_dir=cfg.OUTPUT_DIR, save_to_disk=True, logger=None
        )
        _ = checkpointer.load()
        model = model.eval()

    image = io.imread(impath)
    image_tensor = transform(image.astype(float))[None].to(device)
    meta = {
        "filename": impath,
        "height": image.shape[0],
        "width": image.shape[1],
    }
    torch.onnx.export(model, (image_tensor, [meta]), "hawp.onnx", opset_version=13)
    with torch.no_grad():
        outputs = model(image_tensor, [meta])
        output = {
            "lines_pred": outputs[0],
            "lines_score": outputs[1],
            "juncs_pred": outputs[2],
            "juncs_score": outputs[3],
        }
        output = to_device(output, "cpu")

    sx = meta["width"] / 128
    sy = meta["height"] / 128

    output["lines_pred"][:, 0] *= sx
    output["lines_pred"][:, 1] *= sy
    output["lines_pred"][:, 2] *= sx
    output["lines_pred"][:, 3] *= sy

    output["juncs_pred"][:, 0] *= sx
    output["juncs_pred"][:, 1] *= sy

    lines = output["lines_pred"].numpy()

    if not len(lines):
        return [], []

    scores = output["lines_score"].numpy()
    idx = scores > args.line_threshold
    
    lines = lines[idx]
    scores = scores[idx]
    return lines, scores

if __name__ == "__main__":
    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)
    else:
        cfg.OUTPUT_DIR = "outputs/default"
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR

    logger = setup_logger("hawp", output_dir)
    logger.info(args)
    if args.config_file is not None:
        logger.info("Loaded configuration file {}".format(args.config_file))
    else:
        logger.info("Loaded the default configuration for testing")

    lines, scores = onnx(cfg)
