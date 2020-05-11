import torch
from parsing.config import cfg
from parsing.utils.comm import to_device
from parsing.dataset import build_test_dataset
from parsing.detector import WireframeDetector
from parsing.utils.logger import setup_logger
from parsing.utils.metric_logger import MetricLogger
from parsing.utils.miscellaneous import save_config
from parsing.utils.checkpoint import DetectronCheckpointer
import os
import os.path as osp
import time
import datetime
import argparse
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
parser = argparse.ArgumentParser(description='HAWP Testing')

parser.add_argument("--config-file",
                    metavar="FILE",
                    help="path to config file",
                    type=str,
                    required=True,
                    )

parser.add_argument("--display",
                    default=False,
                    action='store_true')
parser.add_argument("opts",
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER
                    )
args = parser.parse_args()

def test(cfg):
    logger = logging.getLogger("hawp.testing")
    device = cfg.MODEL.DEVICE
    model = WireframeDetector(cfg)
    model = model.to(device)

    test_datasets = build_test_dataset(cfg)
    
    output_dir = cfg.OUTPUT_DIR

    checkpointer = DetectronCheckpointer(cfg,
                                         model,
                                         save_dir=cfg.OUTPUT_DIR,
                                         save_to_disk=True,
                                         logger=logger)
    _ = checkpointer.load()
    model = model.eval()
    for name, dataset in test_datasets:
        results = []
        logger.info('Testing on {} dataset'.format(name))

        for i, (images, annotations) in enumerate(tqdm(dataset)):
            with torch.no_grad():
                output, extra_info = model(images.to(device), annotations)
                output = to_device(output,'cpu')
        
            if args.display:
                im = dataset.dataset.image(i)
                plt.imshow(im)
                lines = output['lines_pred'].numpy()
                scores = output['lines_score'].numpy()
                plt.plot([lines[scores>0.97,0],lines[scores>0.97,2]],
                        [lines[scores>0.97,1],lines[scores>0.97,3]], 'r-')
                plt.show()

            for k in output.keys():
                if isinstance(output[k], torch.Tensor):
                    output[k] = output[k].tolist()
            results.append(output)
        outpath_dataset = osp.join(output_dir,'{}.json'.format(name))
        logger.info('Writing the results of the {} dataset into {}'.format(name,
                    outpath_dataset))
        with open(outpath_dataset,'w') as _out:
            json.dump(results,_out)

if __name__ == "__main__":

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    output_dir = cfg.OUTPUT_DIR
    logger = setup_logger('hawp', output_dir, out_file='test.log')
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))


    test(cfg)

    ### Training



    # import pdb; pdb.set_trace()


