import torch
import random
import numpy as np

from parsing.config import cfg
from parsing.utils.comm import to_device
from parsing.dataset import build_train_dataset
from parsing.detector import WireframeDetector
from parsing.solver import make_lr_scheduler, make_optimizer
from parsing.utils.logger import setup_logger
from parsing.utils.metric_logger import MetricLogger
from parsing.utils.miscellaneous import save_config
from parsing.utils.checkpoint import DetectronCheckpointer
import os
import time
import datetime
import argparse
import logging

class LossReducer(object):
    def __init__(self,cfg):
        # self.loss_keys = cfg.MODEL.LOSS_WEIGHTS.keys()
        self.loss_weights = dict(cfg.MODEL.LOSS_WEIGHTS)
    
    def __call__(self, loss_dict):
        total_loss = sum([self.loss_weights[k]*loss_dict[k] 
        for k in self.loss_weights.keys()])
        
        return total_loss

def train(cfg):
    logger = logging.getLogger("hawp.trainer")
    device = cfg.MODEL.DEVICE
    model = WireframeDetector(cfg)
    model = model.to(device)

    train_dataset = build_train_dataset(cfg)
    
    optimizer = make_optimizer(cfg,model)
    scheduler = make_lr_scheduler(cfg,optimizer)
    
    loss_reducer = LossReducer(cfg)
    
    arguments = {}
    arguments["epoch"] = 0
    max_epoch = cfg.SOLVER.MAX_EPOCH
    arguments["max_epoch"] = max_epoch

    checkpointer = DetectronCheckpointer(cfg,
                                         model,
                                         optimizer,
                                         save_dir=cfg.OUTPUT_DIR,
                                         save_to_disk=True,
                                         logger=logger)

    _ = checkpointer.load()
    start_training_time = time.time()
    end = time.time()

    start_epoch = arguments['epoch']
    epoch_size = len(train_dataset)

    global_iteration = epoch_size*start_epoch

    for epoch in range(start_epoch+1, arguments['max_epoch']+1):
        meters = MetricLogger(" ")
        model.train()
        arguments['epoch'] = epoch

        for it, (images, annotations) in enumerate(train_dataset):
            data_time = time.time() - end
            images = images.to(device)
            annotations = to_device(annotations,device)
            loss_dict, _ = model(images,annotations)
            total_loss = loss_reducer(loss_dict)

            with torch.no_grad():
                loss_dict_reduced = {k:v.item() for k,v in loss_dict.items()}
                loss_reduced = total_loss.item()
                meters.update(loss=loss_reduced, **loss_dict_reduced)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            global_iteration +=1

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_batch = epoch_size*(max_epoch-epoch+1) - it +1
            eta_seconds = meters.time.global_avg*eta_batch
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if it % 20 == 0 or it+1 == len(train_dataset):
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "epoch: {epoch}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}\n",
                        ]
                    ).format(
                        eta=eta_string,
                        epoch=epoch,
                        iter=it,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )


        checkpointer.save('model_{:05d}'.format(epoch))
        scheduler.step()
    
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))

    logger.info(
        "Total training time: {} ({:.4f} s / epoch)".format(
            total_time_str, total_training_time / (max_epoch)
        )
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='HAWP Training')

    parser.add_argument("--config-file",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        required=True,
                        )
    parser.add_argument("--clean",
                        default=False,
                        action='store_true')

    parser.add_argument("--seed",
                        default=2,
                        type=int)
    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER
                        )
    
    
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        if os.path.isdir(output_dir) and args.clean:
            import shutil
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger('hawp', output_dir, out_file='train.log')
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))

    with open(args.config_file,"r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)

    logger.info("Running with config:\n{}".format(cfg))
    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))

    save_config(cfg, output_config_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    train(cfg)

