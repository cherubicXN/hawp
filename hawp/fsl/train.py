import torch
import random
import numpy as np
import os
import os.path as osp
import time
import datetime
import argparse
import logging
import json
from tqdm import tqdm

import hawp
from hawp.base.utils.comm import to_device
from hawp.base.utils.logger import setup_logger
from hawp.base.utils.metric_logger import MetricLogger
from hawp.base.utils.miscellaneous import save_config
from hawp.base.utils.checkpoint import DetectronCheckpointer
from hawp.base.utils.metric_evaluation import TPFP, AP

from hawp.fsl.dataset import build_train_dataset
from hawp.fsl.config import cfg
from hawp.fsl.config.paths_catalog import DatasetCatalog
from hawp.fsl.model.build import build_model
from hawp.fsl.solver import make_lr_scheduler, make_optimizer

AVAILABLE_DATASETS = ('wireframe_test', 'york_test')

def get_output_dir(root, basename):
    timestamp = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    return os.path.join(root,basename,timestamp)

def compute_sap(result_list, annotations_dict, threshold):
    tp_list, fp_list, scores_list = [],[],[]
    n_gt = 0
    for res in result_list:
        filename = res['filename']
        gt = annotations_dict[filename]
        lines_pred = np.array(res['lines_pred'],dtype=np.float32)
        scores = np.array(res['lines_score'],dtype=np.float32)
        sort_idx = np.argsort(-scores)
        
        lines_pred = lines_pred[sort_idx]
        scores = scores[sort_idx]
        # import pdb; pdb.set_trace()
        lines_pred[:,0] *= 128/float(res['width'])
        lines_pred[:,1] *= 128/float(res['height'])
        lines_pred[:,2] *= 128/float(res['width'])
        lines_pred[:,3] *= 128/float(res['height'])

        lines_gt   = np.array(gt['lines'],dtype=np.float32)
        lines_gt[:,0]  *= 128/float(gt['width'])
        lines_gt[:,1]  *= 128/float(gt['height'])
        lines_gt[:,2]  *= 128/float(gt['width'])
        lines_gt[:,3]  *= 128/float(gt['height'])

        tp, fp = TPFP(lines_pred,lines_gt,threshold)
        
        n_gt += lines_gt.shape[0]
        tp_list.append(tp)
        fp_list.append(fp)
        scores_list.append(scores)

    tp_list = np.concatenate(tp_list)
    fp_list = np.concatenate(fp_list)
    scores_list = np.concatenate(scores_list)
    idx = np.argsort(scores_list)[::-1]
    tp = np.cumsum(tp_list[idx])/n_gt
    fp = np.cumsum(fp_list[idx])/n_gt
    rcs = tp
    pcs = tp/np.maximum(tp+fp,1e-9)
    sAP = AP(tp,fp)*100
    return sAP, pcs, rcs

class LossReducer(object):
    def __init__(self,cfg):
        # self.loss_keys = cfg.MODEL.LOSS_WEIGHTS.keys()
        self.loss_weights = dict(cfg.MODEL.LOSS_WEIGHTS)
    
    def __call__(self, loss_dict):
        total_loss = sum([self.loss_weights[k]*loss_dict[k] 
        for k in self.loss_weights.keys()])
        
        return total_loss

def train(cfg, model, train_dataset, optimizer, scheduler, loss_reducer, checkpointer, arguments):
    logger = logging.getLogger("hawp.trainer")
    device = cfg.MODEL.DEVICE
    model = model.to(device)
    start_training_time = time.time()
    end = time.time()

    start_epoch = arguments['epoch']
    num_epochs = arguments['max_epoch'] - start_epoch
    epoch_size = len(train_dataset)
    
    epoch = arguments['epoch'] +1

    total_iterations = num_epochs*epoch_size
    step = 0
    # experiment.clean()
    
    for epoch in range(start_epoch+1, start_epoch+num_epochs+1):
        model.train()            
        loss_meters = MetricLogger(" ")
        aux_meters = MetricLogger(" ")
        sys_meters = MetricLogger(" ")
        # for it, (images, targets, metas) in enumerate(train_dataset):
        for it, (images, annotations) in enumerate(train_dataset):
            data_time = time.time() - end
            images = images.to(device)
            annotations = to_device(annotations,device)
            
            loss_dict, extra_info = model(images,annotations)
            total_loss = loss_reducer(loss_dict)

            with torch.no_grad():
                loss_dict_reduced = {k:v.item() for k,v in loss_dict.items()}
                loss_reduced = total_loss.item()
                loss_meters.update(loss=loss_reduced, **loss_dict_reduced)
                aux_meters.update(**extra_info)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            sys_meters.update(time=batch_time, data=data_time)

            total_iterations -= 1
            step +=1
            
            eta_seconds = sys_meters.time.global_avg*total_iterations
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if it % 20 == 0 or it+1 == len(train_dataset):
                logger.info(
                    "".join(
                        [
                            "eta: {eta} ",
                            "epoch: {epoch} ",
                            "iter: {iter} ",
                            "lr: {lr:.6f} ",
                            "max mem: {memory:.0f}\n",
                            "RUNTIME: {sys_meters}\n",
                            "LOSSES: {loss_meters}\n",
                            "AUXINFO: {aux_meters}\n"
                            "WorkingDIR: {wdir}\n"
                        ]
                    ).format(
                        eta=eta_string,
                        epoch=epoch,
                        iter=it,
                        loss_meters=str(loss_meters),
                        sys_meters=str(sys_meters),
                        aux_meters=str(aux_meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        wdir = cfg.OUTPUT_DIR
                    )
                )

        scheduler.step()

        checkpointer.save('model_{:05d}'.format(epoch))
                
                
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))

    logger.info(
        "Total training time: {} ({:.4f} s / epoch)".format(
            total_time_str, total_training_time / (max_epoch)
        )
    )

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='HAWPv2 Training')

    parser.add_argument("config",
                        # metavar="FILE",
                        help="path to config file",
                        type=str,
                        )

    parser.add_argument('--logdir',required=True, type=str)
    parser.add_argument('--resume',default=None, type=str)
    parser.add_argument("--clean",
                        default=False,
                        action='store_true')
    parser.add_argument("--seed",
                        default=42,
                        type=int)
    
    parser.add_argument('--tf32', default=False, action='store_true', help='toggle on the TF32 of pytorch')
    parser.add_argument('--dtm', default=True, choices=[True, False], help='toggle the deterministic option of CUDNN. This option will affect the replication of experiments')

    args = parser.parse_args()
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.deterministic = args.dtm

    assert args.config.endswith('yaml') or args.config.endswith('yml')
    config_basename = os.path.basename(args.config)
    if config_basename.endswith('yaml'):
        config_basename = config_basename[:-5]
    else:
        config_basename = config_basename[:-4]

    cfg.merge_from_file(args.config)

    output_dir = get_output_dir(args.logdir,config_basename)
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(output_dir)
    

    logger = setup_logger('hawp', output_dir, out_file='train.log')

    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config))

    with open(args.config,"r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)

    logger.info("Running with config:\n{}".format(cfg))
    output_config_path = os.path.join(output_dir, 'config.yaml')
    logger.info("Saving config into: {}".format(output_config_path))
    save_config(cfg, output_config_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    model = build_model(cfg)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

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
    if args.resume:
        state_dict = torch.load(args.resume,map_location='cpu')
        model.load_state_dict(state_dict['model'],strict=False)
        logger.info('loading the pretrained model from {}'.format(args.resume))
        
    train_dataset = build_train_dataset(cfg)
    
    logger.info('epoch size = {}'.format(len(train_dataset)))
    train(cfg, model, train_dataset, optimizer, scheduler, loss_reducer, checkpointer, arguments)    

                                         
    
    import pdb; pdb.set_trace()