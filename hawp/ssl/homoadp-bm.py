import torch
import random 
import numpy as np
import cv2

import hawp
from hawp.base import to_device, setup_logger, MetricLogger, save_config

from hawp.fsl.config import cfg as model_config
from hawp.fsl.solver import make_lr_scheduler, make_optimizer
from kornia.geometry import warp_perspective,transform_points

from .config import Config, load_config
from .datasets import dataset_util
from .datasets.transforms.homographic_transforms import sample_homography
from .models import MODELS


from torch.utils.data import DataLoader
import torch.utils.data.dataloader as torch_loader

from pathlib import Path
from tqdm import tqdm
import argparse, yaml, logging, time, datetime, copy, math, h5py, hashlib
import matplotlib.pyplot as plt

YAML_TEMPLATE = load_config(Path(__file__).parent/'config/exports/template.yaml')

class ArgumentDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Help message formatter which adds default values to argument help.
    Only the name of this class is considered a public API. All the methods
    provided by the class are considered an implementation detail.
    """

    def _get_help_string(self, action):
        if action.default is not None:
            help = super(ArgumentDefaultsHelpFormatter,self)._get_help_string(action)
        else:
            help = action.help
        
        return help

    # def _get_default_metavar_for_optional(self, action):
    #     return action.type.__name__

    # def _get_default_metavar_for_positional(self, action):
    #     return action.type.__name__
def CheckExt(choices):
    class Act(argparse.Action):
        def __call__(self,parser,namespace,fname,option_string=None):
            ext = Path(fname).suffix

            if ext not in choices:
                option_string = '({})'.format(option_string) if option_string else ''
                parser.error("file doesn't end with one of {}{}".format(choices,option_string))
            else:
                setattr(namespace,self.dest,fname)

    return Act

# Get the filename padded with 0.
def get_padded_filename(num_pad, idx):
    file_len = len("%d" % (idx))
    filename = "0" * (num_pad - file_len) + "%d" % (idx)

    return filename

def adjust_border(input_masks, device, margin=3):
    """ Adjust the border of the counts and valid_mask. """
    # Convert the mask to numpy array
    dtype = input_masks.dtype
    input_masks = np.squeeze(input_masks.cpu().numpy(), axis=1)

    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (margin*2, margin*2))
    batch_size = input_masks.shape[0]
    
    output_mask_lst = []
    # Erode all the masks
    for i in range(batch_size):
        output_mask = cv2.erode(input_masks[i, ...], erosion_kernel)

        output_mask_lst.append(
            torch.tensor(output_mask, dtype=dtype, device=device)[None])
    
    # Concat back along the batch dimension.
    output_masks = torch.cat(output_mask_lst, dim=0)
    return output_masks.unsqueeze(dim=1)


def parse_args():
    import sys
    aparser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    group = aparser.add_argument_group('required arguments')
    group.add_argument('--metarch',default='HAWP-heatmap',choices=MODELS.keys())
    group.add_argument('--datacfg',required=True, type=str, help = 'filepath of the data config')
    group.add_argument('--workdir',default=None, type=str, help= 'working directory')
    group.add_argument('--epoch',default=None, required='--workdir' in sys.argv, type=int, help = 'the epoch number')
    group.add_argument('--modelcfg',required='--workdir' not in sys.argv, type=str, help = 'filepath of the model config')
    group.add_argument('-c','--ckpt', required='--workdir' not in sys.argv, type=str, help = 'path of the checkpoint')
    group.add_argument('-d','--dest', required='--workdir' not in sys.argv, type=str, help = 'the path of the exported pseudo labels', action=CheckExt({'.h5'}))

    group_filter = aparser.add_argument_group('line segment filtering options')

    group_filter.add_argument('--min-score','--min_score',default=0.75,type=float, help = 'the minimum score threshold of line segments')
    group_filter.add_argument('--ajk', default=False, action='store_true', help = 'ajk means All Junctions are Kept. If True, the junctions that do not associate with lines are kept')

    group_debug = aparser.add_argument_group('debug options')
    group_debug.add_argument('--display', default=False, action='store_true')
    group_debug.add_argument('--batch-size','--batch_size', default=1, type=int)

    for k in MODELS.keys():
        MODELS[k].cli(aparser)

    
    args = aparser.parse_args()

    if args.workdir is not None:
        args.modelcfg = Path(args.workdir)/'model.yaml'
        ckpt = 'model-{:05d}.pth'.format(args.epoch)
        args.ckpt = Path(args.workdir)/ckpt#('model-{:05d}.pth'.format(args.epoch))
        export_dir = Path(Config.export_dataroot)/Path(args.workdir).name
        export_dir.mkdir(exist_ok=True)


        # import pdb; pdb.set_trace()
        # args.dest = export_dir/'model-{:05d}.h5'.format(args.epoch)
        cmd = ''
        cmd += ' '.join(sys.argv)
        cmd += '\n'
        cmd += '# python -m sslib.export-debug \\\n'
        for key, val in args.__dict__.items():
            if key in [x.dest for x in group_debug._group_actions]:
                continue
            cmd += '# --{} {}'.format(key,val)
            cmd += '\\\n'

        hash = hashlib.md5(cmd.encode('utf-8')).hexdigest()

        args.dest = export_dir/'{}-model-{:05d}.h5'.format(hash, args.epoch)
        with open(export_dir/'{}-model-{:05d}.sh'.format(hash,args.epoch),'w') as writer:
            writer.write(cmd)
    

    for k in MODELS.keys():
        MODELS[k].configure(args)
    return args

def export_homography_adaptation(args, model,
    dataset_cfg, export_dataset_mode, device, *, logger):
    output_path = Path(args.dest)

    supported_modes = ["train", "test"]

    if not export_dataset_mode in supported_modes:
        raise ValueError(
            "[Error] The specified export_dataset_mode is not supported.")

    dataset_cfg['augmentation']['photometric']['enable'] = False
    dataset_cfg['augmentation']['homographic']['enable'] = False

    homography_cfg = dataset_cfg.get("homography_adaptation", None)
    if homography_cfg is None:
        raise ValueError(
            "[Error] Empty homography_adaptation entry in config.")
    export_dataset, collate_fn = dataset_util.get_dataset(export_dataset_mode, dataset_cfg)
    export_loader = DataLoader(export_dataset, batch_size=args.batch_size,
                               num_workers=8,
                               shuffle=False, pin_memory=False,
                               collate_fn=collate_fn)
    
    

    YAML_TEMPLATE['gt_source_train'] = str(args.dest)
    YAML_TEMPLATE['dataset_name'] = dataset_cfg['dataset_name']

    with open(args.dest.with_suffix('.yaml'), 'w') as f:
        yaml.safe_dump(YAML_TEMPLATE, f)
    
    with h5py.File(output_path, "w", libver="latest", swmr=True) as f:
        num_lines_list = []
        for filename_idx, data in enumerate(tqdm(export_loader)):
            valid_mask = data["valid_mask"]
            outputs_batch = homography_adaptation(args,data['image'].cuda(),model,homography_cfg)
            batch_idx = 0
            
            for batch_idx, outputs in enumerate(outputs_batch):
                output_data = {
                        # "image": data['image'].cpu().numpy().transpose(0, 2, 3, 1)[batch_idx, ...],
                        "junctions": outputs['junctions'].cpu().numpy(),
                        "line_map": outputs['line_map'].float().cpu().numpy(),
                    }
                num_lines_list.append(outputs['line_map'].triu().sum().item())

                num_pad = math.ceil(math.log10(len(export_loader)*args.batch_size)) + 1
                output_key = get_padded_filename(num_pad, filename_idx*args.batch_size+batch_idx)
                f_group = f.create_group(output_key)
                for key, output_data in output_data.items():
                    f_group.create_dataset(key, data=output_data,
                                        compression="gzip")

            if filename_idx%20 == 0:
                # logger.info('Avg Lines: {}'.format(sum(num_lines_list)/len(num_lines_list)))
                logger.info('h5-Dest: {}'.format(args.dest))


    logger.info('Label Generation Done!')
    logger.info('Label path: {}'.format(args.dest))
    logger.info('YAML file path: {}'.format(args.dest.with_suffix('.yaml')))
    logger.info('You can use {} to train a new model'.format(args.dest.with_suffix('.yaml')))
    

def homography_adaptation(args,input_images, model, homography_cfg):
    device = next(model.parameters()).device
    batch_size, _, H, W = input_images.shape
    num_iter = homography_cfg["num_iter"]
    junc_probs = torch.zeros([batch_size, num_iter, H, W], device=device)
    junc_counts = torch.zeros([batch_size, 1, H, W], device=device)
    heatmap_probs = torch.zeros([batch_size, num_iter, H, W], device=device)
    heatmap_counts = torch.zeros([batch_size, 1, H, W], device=device)
    margin = homography_cfg["valid_border_margin"]

    homography_cfg_no_artifacts = copy.copy(homography_cfg["homographies"])
    homography_cfg_no_artifacts["allow_artifacts"] = False

    meta = [{
        'filename': '',
        'width': W,
        'height': H,
    }]*batch_size

    
    H_tensor_list = []
    H_inv_tensor_list = []

    for idx in range(num_iter):
        if idx <= num_iter // 5:
            # Ensure that 20% of the homographies have no artifact
            H_mat_lst = [sample_homography(
                [H,W], **homography_cfg_no_artifacts)[0][None]
                         for _ in range(batch_size)]
        else:
            H_mat_lst = [sample_homography(
                [H,W], **homography_cfg["homographies"])[0][None]
                         for _ in range(batch_size)]
        
        H_mats = np.concatenate(H_mat_lst, axis=0)
        H_tensor = torch.tensor(H_mats, dtype=torch.float, device=device)
        H_inv_tensor = torch.inverse(H_tensor)

        H_tensor_list.append(H_tensor)
        H_inv_tensor_list.append(H_inv_tensor)

    H_tensor_list = torch.stack(H_tensor_list,dim=1)
    H_inv_tensor_list = torch.stack(H_inv_tensor_list,dim=1)
    # images_warped = warp_perspective(input_images.repeat((H_tensor_list.shape[0],1,1,1)),H_tensor_list,(H,W),mode='bilinear')
    images_warped = warp_perspective(
        input_images.repeat((1,num_iter,1,1)).reshape(-1,1,H,W), 
        H_tensor_list.flatten(0,1), (H,W), mode='bilinear')

    masks_heatmap_warped = warp_perspective(
        torch.ones([batch_size*num_iter, 1, H, W], device=device), 
        H_tensor_list.flatten(0,1), (H, W), mode="nearest"
    )
    heatmap_counts = warp_perspective(masks_heatmap_warped, H_inv_tensor_list.flatten(0,1), (H, W), mode="nearest")
    heatmap_counts = heatmap_counts.reshape(batch_size,num_iter,H,W)
    
    with torch.no_grad():
        heatmap_probs_warped = model.compute_heatmaps(images_warped,meta)
    

    heatmap_probs = warp_perspective(heatmap_probs_warped, H_inv_tensor_list.flatten(0,1),(H,W),mode='bilinear')
    heatmap_probs = heatmap_probs.reshape(batch_size,num_iter,H,W)

    heatmap_counts = torch.sum(heatmap_counts,dim=1,keepdim=True)
    heatmaps_averaged = heatmap_probs.sum(dim=1,keepdim=True)/heatmap_counts
    outputs_batch, _ = model.detect_with_heatmaps(input_images, meta, heatmaps=heatmaps_averaged,min_score=args.min_score)
    # import pdb; pdb.set_trace()

    ret = []
    for i,outputs in enumerate(outputs_batch):
        lines = outputs['lines_pred']
        scores = outputs['lines_score']
        junctions = outputs['juncs_pred']

        if junctions.shape[0]>0 and lines.shape[0]>0:
            dis_1,vertices_1 = torch.sum((junctions[:,None] - lines[None,:,:2])**2,dim=-1).min(dim=0)
            dis_2,vertices_2 = torch.sum((junctions[:,None] - lines[None,:,2:])**2,dim=-1).min(dim=0)

            if args.ajk == False:
                junctions_mask = torch.zeros((junctions.shape[0]),device=device)
                junctions_mask[vertices_1] = 1
                junctions_mask[vertices_2] = 1

                junctions = junctions[junctions_mask>0]
                
                dis_1,vertices_1 = torch.sum((junctions[:,None] - lines[None,:,:2])**2,dim=-1).min(dim=0)
                dis_2,vertices_2 = torch.sum((junctions[:,None] - lines[None,:,2:])**2,dim=-1).min(dim=0)
            
            graph = torch.zeros((junctions.shape[0],junctions.shape[0]),device=device)
            
            graph[vertices_1,vertices_2] = 1
            graph[vertices_2,vertices_1] = 1

        else:
            graph = torch.zeros((junctions.shape[0],junctions.shape[0]),device=device)

        if args.display:
            _ = junctions[graph.triu().nonzero()].reshape(-1,4).cpu().numpy()
            plt.imshow(input_images[i,0].cpu())
            plt.plot([_[:,0],_[:,2]],[_[:,1],_[:,3]],'r-')
            plt.plot(junctions[:,0].cpu().numpy(),junctions[:,1].cpu().numpy(),'b.')
            plt.show()

        ret.append(
            {
                'junctions': junctions, 
                'line_map':  graph,
            }
        )

    return ret
    
    # import pdb; pdb.set_trace()



def main():

    args = parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    dataset_config = load_config(args.datacfg)

    model_config.merge_from_file(args.modelcfg)

    model = MODELS[args.metarch](model_config, gray_scale=dataset_config.get('gray_scale',True))
    model = model.eval().cuda()
    state_dict = torch.load(args.ckpt,map_location='cpu')
    model.load_state_dict(state_dict)

    logger = setup_logger('hawp.export', args.dest.parent, args.dest.with_suffix('.log').name, json_format=True)
    export_homography_adaptation(args, model, dataset_config, 'train',device='cuda', logger=logger)

if __name__ == "__main__":
    main()