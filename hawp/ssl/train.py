import torch
import random 
import numpy as np

import hawp
from hawp.base.utils.comm import to_device
from hawp.base.utils.logger import setup_logger
from hawp.base.utils.metric_logger import MetricLogger
from hawp.base.utils.miscellaneous import save_config

from hawp.fsl.solver import make_lr_scheduler, make_optimizer
from hawp.fsl.config import cfg as model_config

from hawp.ssl.config import Config, load_config
from hawp.ssl.datasets import dataset_util
from hawp.ssl.models import HAWP

# from .config import Config, load_config
# from .datasets import dataset_util
# from .models import HAWP


from torch.utils.data import DataLoader
import torch.utils.data.dataloader as torch_loader


from pathlib import Path
import argparse
import yaml
import logging
import time
import datetime

class LossReducer(object):
    def __init__(self,cfg):
        # self.loss_keys = cfg.MODEL.LOSS_WEIGHTS.keys()
        self.loss_weights = dict(cfg.MODEL.LOSS_WEIGHTS)
    
    def __call__(self, loss_dict):
        total_loss = sum([self.loss_weights[k]*loss_dict[k] 
        for k in self.loss_weights.keys()])
        
        return total_loss
        

def parse_args():
    aparser = argparse.ArgumentParser()

    aparser.add_argument('--datacfg',required=True, type=str, help = 'filepath of the data config')
    aparser.add_argument('--modelcfg',required=True, type=str, help = 'filepath of the model config')
    aparser.add_argument('--name', required=True, type=str, help='the name of experiment')
    aparser.add_argument('--pretrained', default=None, type=str, help='the pretrained model')
    aparser.add_argument('--overwrite', default=False, action='store_true', help='[Caution!] the option to overwrite an existed experiment')
    aparser.add_argument('--tf32', default=False, action='store_true', help='toggle on the TF32 of pytorch')
    aparser.add_argument('--dtm', default=True, choices=[True, False], help='toggle the deterministic option of CUDNN. This option will affect the replication of experiments')

    group = aparser.add_argument_group('training recipe')
    group.add_argument('--batch-size', default=16, type=int, help='the batch size of training')
    group.add_argument('--num-workers', default=8, type=int, help='the number of workers for training')
    group.add_argument('--base-lr', default=4e-4, type=float, help='the initial learning rate')
    group.add_argument('--steps', default=[25], type=int,nargs='+', help = 'the steps of the scheduler')
    group.add_argument('--gamma', default=0.1, type=float, help = 'the lr decay factor')
    group.add_argument('--epochs',default=30, type=int, help = 'the number of epochs for training')
    group.add_argument('--seed',default=None, type=int, help = 'the random seed for training')

    group.add_argument('--iterations',default=None, type=int, help = 'the number of training iterations')
    args = aparser.parse_args()
    return args

def main():

    args = parse_args()
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.deterministic = args.dtm
    dataset_config = load_config(args.datacfg)

    model_config.merge_from_file(args.modelcfg)

    train_dataset, train_collate_fn = dataset_util.get_dataset('train', dataset_config)
    
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True, pin_memory=True,
                              collate_fn=train_collate_fn)
    if 'generation' in dataset_config:
        random_seed = dataset_config['generation']['random_seed']
    elif 'random_seed' in dataset_config:
        random_seed = dataset_config['random_seed']
    if args.seed is not None:
        random_seed = args.seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    
    #torch.use_deterministic_algorithms(True)

    output_dir = Path(Config.EXP_PATH)/args.name
    
    output_dir.mkdir(exist_ok = args.overwrite)

    logger = setup_logger('hawp', output_dir, out_file='train.log')

    logger.info(args)

    logger.info('Dataset cfg')
    logger.info(dataset_config)

    save_config(model_config, output_dir/'model.yaml')
    
    with open(output_dir/'dataset.yaml','w') as f:
        yaml.dump(dataset_config,f)

    model = HAWP(model_config, gray_scale=dataset_config.get('gray_scale',True))
    
    if args.pretrained:
        state_dict = torch.load(args.pretrained,map_location='cpu')
        model.load_state_dict(state_dict,strict=False)

    model_config.merge_from_list(
        ['SOLVER.BASE_LR', args.base_lr,
         'SOLVER.STEPS', args.steps,
         'SOLVER.GAMMA', args.gamma,
         'SOLVER.MAX_EPOCH', args.epochs,
        ]
    )
    logger.info('Model cfg')
    logger.info(model_config)
    
    loss_reducer = LossReducer(model_config)
    # loss_reducer = loss_reducer.to('cuda')

    optimizer = make_optimizer(model_config,model)
    scheduler = make_lr_scheduler(model_config,optimizer)
    

    arguments = {}
    arguments["epoch"] = 0
    max_epoch = model_config.SOLVER.MAX_EPOCH
    arguments["max_epoch"] = max_epoch

    train(model, train_loader, optimizer, scheduler, loss_reducer, arguments, output_dir)

def train(model, train_loader, optimizer, scheduler, loss_reducer, arguments, output_dir):
    logger = logging.getLogger("hawp.trainer")
    device = model_config.MODEL.DEVICE
    model = model.to(device)
    start_training_time = time.time()
    end = time.time()

    start_epoch = arguments['epoch']
    num_epochs = arguments['max_epoch'] - start_epoch
    epoch_size = len(train_loader)
    
    epoch = arguments['epoch'] +1

    total_iterations = num_epochs*epoch_size
    step = 0
    # experiment.clean()
    
    model.train()       
    for epoch in range(start_epoch+1, start_epoch+num_epochs+1):
        loss_meters = MetricLogger(" ")
        aux_meters = MetricLogger(" ")
        sys_meters = MetricLogger(" ")
        
        for it, data in enumerate(train_loader):
            data_time = time.time() - end
            images = data.pop('image').cuda()
            annotations = to_device(data,images.device)  
            annotations['batch_size'] = images.shape[0]
            annotations['width'] = images.shape[-1]
            annotations['height'] = images.shape[-2]
            annotations['stride'] = 4

            loss_dict, extra_info = model(images,annotations)
            total_loss = loss_reducer(loss_dict)

            with torch.no_grad():
                loss_dict_reduced = {k:float(v) for k,v in loss_dict.items()}
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

            if it % 20 == 0 or it+1 == len(train_loader):
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
                        wdir = output_dir
                    )
                )
        scheduler.step()
        torch.save(model.state_dict(),output_dir/'model-{:05d}.pth'.format(epoch))
        logger.info('saving the state dict into {}'.format(output_dir/'model-{:05d}.pth'.format(epoch)))

if __name__ == "__main__":
    main()