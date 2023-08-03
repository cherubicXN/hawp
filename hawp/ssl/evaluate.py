import torch
import random 
import numpy as np

import hawp
from hawp.base import to_device, setup_logger, MetricLogger, save_config
from hawp.fsl.config import cfg as model_config
from scipy.ndimage import binary_erosion

from hawp.ssl.config import Config, load_config
from hawp.ssl.datasets import dataset_util
from hawp.ssl.datasets.transforms.homographic_transforms import compute_valid_mask
from hawp.ssl.misc.geometry_utils import get_overlap_orth_line_dist
from hawp.ssl.models import MODELS


from torch.utils.data import DataLoader
import torch.utils.data.dataloader as torch_loader
from kornia.geometry.transform import resize
from pathlib import Path
import argparse
import yaml
import logging
import time
import datetime

def compute_distances(line_segments_anchor, line_segments_cand, dist_tolerance_lst, distance_metric="sAP", group_num=1000):
    if not distance_metric in ["sAP", "sAP_square", "orthogonal_distance"]:
        raise ValueError("[Error] The specified distance metric is not supported.")
    
    # Compute distance matrix
    if distance_metric == "sAP" or distance_metric == "sAP_square":
        num_anchor_seg = line_segments_anchor.shape[0]
        min_dist_lst = []
        if num_anchor_seg > group_num:
            num_iter = math.ceil(num_anchor_seg / group_num)
            for iter_idx in range(num_iter):
                if iter_idx == num_iter - 1:
                    if distance_metric == "sAP":
                        diff = (((line_segments_anchor[iter_idx*group_num:, None, :, None] - line_segments_cand[None, :, None]) ** 2).sum(-1)) ** 0.5
                    else:
                        diff = (((line_segments_anchor[iter_idx*group_num:, None, :, None] - line_segments_cand[None, :, None]) ** 2).sum(-1))
                    diff = np.minimum(
                        diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
                    )
                else:
                    if distance_metric == "sAP":
                        diff = (((line_segments_anchor[iter_idx*group_num:(iter_idx+1)*group_num, None, :, None] - line_segments_cand[:, None]) ** 2).sum(-1)) ** 0.5
                    else:
                        diff = (((line_segments_anchor[iter_idx*group_num:(iter_idx+1)*group_num, None, :, None] - line_segments_cand[:, None]) ** 2).sum(-1))
                    diff = np.minimum(
                        diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
                    )
                # Compute reference to target correctness
                try:
                    anchor_cand_min_dist_ = np.min(diff, 1)
                except:
                    # if diff is empty
                    anchor_cand_min_dist_ = np.ones([diff.shape[0], 1]) * (dist_tolerance_lst[-1] + 100.)
                min_dist_lst.append(anchor_cand_min_dist_)
            anchor_cand_min_dist = np.concatenate(min_dist_lst)
        else:
            if distance_metric == "sAP":
                diff = (((line_segments_anchor[:, None, :, None] - line_segments_cand[None, :, None]) ** 2).sum(-1)) ** 0.5
            else:
                diff = (((line_segments_anchor[:, None, :, None] - line_segments_cand[None, :, None]) ** 2).sum(-1))
            diff = np.minimum(
                diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
            )
            # Compute reference to target correctness
            try:
                anchor_cand_min_dist = np.min(diff, 1)
            except:
                # if diff is empty
                anchor_cand_min_dist = np.ones([diff.shape[0], 1]) * (dist_tolerance_lst[-1] + 100.)

    elif distance_metric == "orthogonal_distance":
        if 0 in line_segments_anchor.shape or 0 in line_segments_cand.shape:
            if 0 in line_segments_cand.shape:
                diff = np.ones([line_segments_anchor.shape[0], 1]) * (dist_tolerance_lst[-1] + 100.)
            else:
                diff = np.ones([1, 1]) * (dist_tolerance_lst[-1] + 100.)
        else:
            diff = get_overlap_orth_line_dist(
                line_segments_anchor,
                line_segments_cand,
                min_overlap=0.5
            )

        # Compute reference to target correctness
        try:
            anchor_cand_min_dist = np.min(diff, 1)
        except:
            # if diff is empty
            anchor_cand_min_dist = np.ones([diff.shape[0], 1]) * (dist_tolerance_lst[-1] + 100.)
        # import ipdb; ipdb.set_trace()
    
    return anchor_cand_min_dist

def compute_metrics_v2(
        line_segments_ref, 
        line_segments_target, 
        valid_mask, 
        H_mat, 
        image_size,
        dist_tolerance_lst,
        distance_metric="sAP",
        erode_border=False,
        erode_border_margin=2,
    ):
    """
    line_segments_ref: Nx2x2 array.
    line_segments_target: Nx2x2 array.
    valid_mask: 2D mask (same size as the image)
    H_mat: the 3x3 array containing the homography matrix.
    image_size: list containing [H, W].
    dist_tolerance_lst: list of all distance tolerances of interest.
    distance_metric: "sAP" or "orthogonal_distance".
    """
    # Verify the shapes
    if (not isinstance(line_segments_ref, np.ndarray)) or len(line_segments_ref.shape) < 3:
        raise ValueError("[Error] line_segments_ref should be an array with shape Nx2x2")
    if (not isinstance(line_segments_target, np.ndarray)) or len(line_segments_target.shape) < 3:
        raise ValueError("[Error] line_segments_target should be an array with shape Nx2x2")
    if not (len(H_mat.shape) == 2 and H_mat.shape[0] == 3 and H_mat.shape[1] == 3):
        raise ValueError("[Error] H_mat should be a 3x3 array")
    
    # Check the distance_metric to use
    supported_metrics = ["sAP", "orthogonal_distance", "sAP_square"]
    if not distance_metric in supported_metrics:
        raise ValueError(f"[Error] The specified distnace metric is not in supported metrics {supported_metrics}.")
    
    # Exclude the segments with endpoints in the boundary margin at the beginning
    if erode_border:
        # Compute the eroded valid masks (ref + target)
        if erode_border_margin < 1:
            raise ValueError("[Error] The erosion margin must be >= 1")
        ref_valid_mask = np.ones(image_size, dtype=np.float32)
        ref_valid_mask = binary_erosion(ref_valid_mask, iterations=erode_border_margin).astype(np.float32)
        target_valid_mask = binary_erosion(valid_mask, iterations=erode_border_margin).astype(np.float32)

        # Exclude segment based on the valid masks
        ref_valid_region_mask1 = ref_valid_mask[line_segments_ref[:, 0, 0].astype(np.int64), line_segments_ref[:, 0, 1].astype(np.int64)] == 1.
        ref_valid_region_mask2 = ref_valid_mask[line_segments_ref[:, 1, 0].astype(np.int64), line_segments_ref[:, 1, 1].astype(np.int64)] == 1.
        ref_valid_region_mask = ref_valid_region_mask1 * ref_valid_region_mask2
        line_segments_ref = line_segments_ref[ref_valid_region_mask, :]
        target_valid_region_mask1 = target_valid_mask[line_segments_target[:, 0, 0].astype(np.int64), line_segments_target[:, 0, 1].astype(np.int64)] == 1.
        target_valid_region_mask2 = target_valid_mask[line_segments_target[:, 1, 0].astype(np.int64), line_segments_target[:, 1, 1].astype(np.int64)] == 1.
        target_valid_region_mask = target_valid_region_mask1 * target_valid_region_mask2
        line_segments_target = line_segments_target[target_valid_region_mask, :]
    else:
        ref_valid_mask = np.ones(image_size, dtype=np.float32)
        target_valid_mask = valid_mask
    
    # Exclude the target segments with endpoints in the clip border
    target_clip_valid_mask = np.ones(image_size, dtype=np.float32)
    if erode_border_margin > 0:
        target_clip_valid_mask = binary_erosion(target_clip_valid_mask, iterations=erode_border_margin).astype(np.float32)
    target_valid_mask = target_valid_mask * target_clip_valid_mask

    line_segments_target = line_segments_target.clip(min=0,max=511)
    
    target_valid_region_mask1 = target_valid_mask[line_segments_target[:, 0, 0].astype(np.int64), line_segments_target[:, 0, 1].astype(np.int64)] == 1.
    target_valid_region_mask2 = target_valid_mask[line_segments_target[:, 1, 0].astype(np.int64), line_segments_target[:, 1, 1].astype(np.int64)] == 1.
    target_valid_region_mask = target_valid_region_mask1 * target_valid_region_mask2
    line_segments_target = line_segments_target[target_valid_region_mask, :]

    # Compute repeatability
    num_segments_ref = line_segments_ref.shape[0]
    num_segments_target = line_segments_target.shape[0]

    # Warp ref line segments to target
    # Convert to xy format => homogeneous
    line_ref_homo = np.concatenate([np.flip(line_segments_ref, -1), np.ones([num_segments_ref, 2, 1])], axis=-1)
    line_ref_warped = line_ref_homo.dot(H_mat.T)
    # Normalize => back to HW format
    line_segments_ref_warped = np.flip(line_ref_warped[:, :, :2] / line_ref_warped[:, :, 2:], -1)

    # Filter out the out-of-border segments in target view (True => keep)
    boundary_mask = np.sum(np.sum((line_segments_ref_warped < 0).astype(np.int64), axis=-1), axis=-1)
    boundary_mask += np.sum((line_segments_ref_warped[:, :, 0] >= image_size[0]-1).astype(np.int64), axis=-1)
    boundary_mask += np.sum((line_segments_ref_warped[:, :, 1] >= image_size[1]-1).astype(np.int64), axis=-1)
    boundary_mask = (boundary_mask == 0)
    line_segments_ref_warped = line_segments_ref_warped[boundary_mask, :]
    # Filter out the out of valid_mask segments in taget view (True => keep)
    valid_region_mask1 = target_valid_mask[line_segments_ref_warped[:, 0, 0].astype(np.int64), line_segments_ref_warped[:, 0, 1].astype(np.int64)] == 1.
    valid_region_mask2 = target_valid_mask[line_segments_ref_warped[:, 1, 0].astype(np.int64), line_segments_ref_warped[:, 1, 1].astype(np.int64)] == 1.
    valid_region_mask = valid_region_mask1 * valid_region_mask2
    line_segments_ref_warped = line_segments_ref_warped[valid_region_mask, :]
    # Perform the filtering on original segments (2 stage)
    line_segments_ref_valid = line_segments_ref[boundary_mask, :]
    line_segments_ref_valid = line_segments_ref_valid[valid_region_mask, :, :]
    # Valid number of segments in ref
    num_valid_segments_ref = line_segments_ref_valid.shape[0]

    # Warp target line segments to ref
    line_target_homo = np.concatenate([np.flip(line_segments_target, -1), np.ones([num_segments_target, 2, 1])], axis=-1)
    line_target_warped = line_target_homo.dot(np.linalg.inv(H_mat.T))
    line_segments_target_warped = np.flip(line_target_warped[:, :, :2] / line_target_warped[:, :, 2:], -1)
    # Filter out the out-of-border segments in ref view (True => keep)
    boundary_mask = np.sum(np.sum((line_segments_target_warped < 0).astype(np.int64), axis=-1), axis=-1)
    boundary_mask += np.sum((line_segments_target_warped[:, :, 0] >= image_size[0]-1).astype(np.int64), axis=-1)
    boundary_mask += np.sum((line_segments_target_warped[:, :, 1] >= image_size[1]-1).astype(np.int64), axis=-1)
    boundary_mask = (boundary_mask == 0)
    line_segments_target_warped = line_segments_target_warped[boundary_mask, :]
    # Filter out the out of valid_mask segments in taget view (True => keep)
    valid_region_mask1 = ref_valid_mask[line_segments_target_warped[:, 0, 0].astype(np.int64), line_segments_target_warped[:, 0, 1].astype(np.int64)] == 1.
    valid_region_mask2 = ref_valid_mask[line_segments_target_warped[:, 1, 0].astype(np.int64), line_segments_target_warped[:, 1, 1].astype(np.int64)] == 1.
    valid_region_mask = valid_region_mask1 * valid_region_mask2
    line_segments_target_warped = line_segments_target_warped[valid_region_mask, :]
    # Directly assign
    line_segments_target_valid = line_segments_target_warped
    # Valid number of segments in ref
    num_valid_segments_target = line_segments_target_valid.shape[0]
    

    # Compute closest segments in taget segments for each ref segment.
    ref_target_min_dist = compute_distances(
        line_segments_ref_valid, line_segments_target_valid,
        dist_tolerance_lst, distance_metric,
        group_num=1000
    )
    
    ref_target_correctness_lst = []
    ref_target_loc_error_lst = []
    for dist_tolerance in dist_tolerance_lst:
        # Compute the correctness for repeatability
        ref_correct_mask = ref_target_min_dist <= dist_tolerance
        ref_target_correctness = np.sum((ref_correct_mask).astype(np.int64))
        ref_target_correctness_lst.append(ref_target_correctness)

        # Compute the localization error
        ref_target_loc_error = ref_target_min_dist[ref_correct_mask]
        ref_target_loc_error_lst.append(ref_target_loc_error)
    
    # Compute closest segments in taget segments for each ref segment.
    target_ref_min_dist = compute_distances(
        line_segments_target_valid, line_segments_ref_valid,
        dist_tolerance_lst, distance_metric,
        group_num=1000
    )
    
    target_ref_correctness_lst = []
    target_ref_loc_error_lst = []
    for dist_tolerance in dist_tolerance_lst:
        # Compute the correctness for repeatability
        traget_correct_mask = target_ref_min_dist <= dist_tolerance
        target_ref_correctness = np.sum((traget_correct_mask).astype(np.int64))
        target_ref_correctness_lst.append(target_ref_correctness)

        # Compute the localization error
        target_ref_loc_error = target_ref_min_dist[traget_correct_mask]
        target_ref_loc_error_lst.append(target_ref_loc_error)
    
    # Record the final correctness
    repeatability_results = {}
    loc_error_results = {}
    for i, dist in enumerate(dist_tolerance_lst):
        # Compute the final repeatability
        # import ipdb; ipdb.set_trace()
        correctness = (ref_target_correctness_lst[i] + target_ref_correctness_lst[i]) / (num_valid_segments_ref + num_valid_segments_target)
        if np.isnan(correctness) or np.isinf(correctness):
            correctness = 0
        repeatability_results[dist] = correctness

        # Compute the final localization error
        # loc_error_lst = np.concatenate([ref_target_loc_error_lst[i], 
        #                                 target_ref_loc_error_lst[i]])
        # Only compute over target segments
        # import ipdb; ipdb.set_trace()
        loc_error_lst = target_ref_loc_error_lst[i]
        if 0 in loc_error_lst.shape:
            loc_error = 0
        else:
            loc_error = np.mean(loc_error_lst)
        loc_error_results[dist] = loc_error
    
    # Return the processed segments
    line_segments_filtered = {
        "line_segments_ref": line_segments_ref_valid,
        "line_segments_target": line_segments_target_valid
    }
    
    return repeatability_results, loc_error_results, line_segments_filtered

def parse_args():
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--metarch', default='HAWP',choices=MODELS.keys())
    parser.add_argument('--workdir',default=None,type=str, help = 'working directory')
    parser.add_argument('--epoch',default=None, required='--workdir' in sys.argv, type=int, help = 'the epoch number')
    parser.add_argument('--datacfg',default='sslib/config/wireframe_official_gt_config.yaml',type=str,help='data config')
    parser.add_argument('--modelcfg',required='--workdir' not in sys.argv, type=str, help = 'filepath of the model config')
    parser.add_argument('--ckpt', required='--workdir' not in sys.argv, type=str, help='the path for loading checkpoints')
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--verbose',default=False, action='store_true')
    parser.add_argument('--width', default=512, type=int)
    parser.add_argument('--height', default=512, type=int)
    # parser.add_argument('--metric',default='all', choices=['sAP','orth','all'])
    parser.add_argument('--nimages',default=-1, type=int)

    for k in MODELS.keys():
        MODELS[k].cli(parser)

    args = parser.parse_args()

    if args.workdir is not None:
        args.modelcfg = Path(args.workdir)/'model.yaml'
        ckpt = 'model-{:05d}.pth'.format(args.epoch)
        args.ckpt = Path(args.workdir)/ckpt#('model-{:05d}.pth'.format(args.epoch))
    
    for k in MODELS.keys():
        MODELS[k].configure(args)

    return args
if __name__ == "__main__":
    
    args = parse_args()
    
    dataset_cfg_path = args.datacfg

    model_cfg_path = args.modelcfg

    weight_path = args.ckpt

    model_config.merge_from_file(model_cfg_path)

    model = MODELS[args.metarch](model_config, gray_scale=True)
    state_dict = torch.load(weight_path,map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to('cuda').eval()
    model.num_residuals = 2

    ckpt_path = Path(args.ckpt)
    if args.verbose:
        logger = setup_logger('hawp.evaluation',ckpt_path.parent,ckpt_path.name+'.evaluation-{:.2f}.json'.format(args.threshold),json_format=True)
    else:
        logger = setup_logger('hawp.evaluation',ckpt_path.parent,'evaluation-{:.2f}.json'.format(args.threshold),json_format=True)
    logger.info(args)
    with open(dataset_cfg_path, "r") as f:
        dataset_cfg = yaml.safe_load(f)
        dataset_cfg['preprocessing']['resize'] = [args.height, args.width]

    dataset, _ = dataset_util.get_dataset("test", dataset_cfg)
    num_datapoints = len(dataset.rep_eval_datapoints['viewpoint'])

    device = next(model.parameters()).device

    meta = {
            'width': args.width,
            'height':args.height,
            'filename': ''
        }

    rep_results_list_structural = []
    loc_results_list_structural = []

    rep_results_list_orthogonal = []
    loc_results_list_orthogonal = []

    num_lines = []
    from tqdm import tqdm
    for idx in tqdm(range(num_datapoints)):
        # if idx % 5 == 0:o
        data = dataset.get_rep_eval_data('viewpoint', idx)
        ref_image = data["ref_image"][None, ...].to(device)
        tgt_image = data["target_image"][None, ...].to(device)
        H_mat = data["homo_mat"]
        image_size = list(ref_image.shape[2:])

        valid_mask = compute_valid_mask(image_size, H_mat, -2)

        with torch.no_grad():
            outputs_ref, _ = model(resize(ref_image,(512,512)),[meta])
            outputs_tgt, _ = model(resize(tgt_image,(512,512)),[meta])
            # outputs_ref, _ = model(ref_image,[meta])
            # outputs_tgt, _ = model(tgt_image,[meta])
        
        lines_ref = outputs_ref['lines_pred']

        lines_tgt = outputs_tgt['lines_pred']

        scores_ref = outputs_ref['lines_score']
        scores_tgt = outputs_tgt['lines_score']

        lines_ref = lines_ref[scores_ref>args.threshold]
        lines_tgt = lines_tgt[scores_tgt>args.threshold]

        lines_ref = lines_ref.reshape(-1,2,2).cpu().numpy()
        lines_tgt = lines_tgt.reshape(-1,2,2).cpu().numpy()

        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(16,8))
        # plt.subplot(1,2,1)
        # plt.imshow(ref_image[0,0].cpu().numpy())
        # plt.plot([lines_ref[:,0,0],lines_ref[:,1,0]],[lines_ref[:,0,1],lines_ref[:,1,1]],'r-')
        # plt.subplot(1,2,2)
        # plt.imshow(tgt_image[0,0].cpu().numpy())
        # plt.plot([lines_tgt[:,0,0],lines_tgt[:,1,0]],[lines_tgt[:,0,1],lines_tgt[:,1,1]],'r-')
        # plt.show()
        lines_ref = np.flip(lines_ref,-1)
        lines_tgt = np.flip(lines_tgt,-1)

        dist_tolerance_lst = [5]
        erode_border = False
        erode_border_margin = 2
        
        # distance_metric = 'sAP'
        rep_results, loc_results, filtered  = compute_metrics_v2(lines_ref, lines_tgt, valid_mask, H_mat, image_size,dist_tolerance_lst,'sAP',erode_border,erode_border_margin)

        num_lines.append(lines_ref.shape[0])
        num_lines.append(lines_tgt.shape[0])

        rep_results_list_structural.append(rep_results[5])
        loc_results_list_structural.append(loc_results[5])

        rep_results, loc_results, filtered  = compute_metrics_v2(lines_ref, lines_tgt, valid_mask, H_mat, image_size,dist_tolerance_lst,'orthogonal_distance',erode_border,erode_border_margin)

        rep_results_list_orthogonal.append(rep_results[5])
        loc_results_list_orthogonal.append(loc_results[5])
        if args.verbose and idx%20==0:
            logger.info('evaluation',extra={
                'step': idx,
                'rep-5 (S)': sum(rep_results_list_structural)/len(rep_results_list_structural),
                'loc-5 (S)': sum(loc_results_list_structural)/len(loc_results_list_structural),
                'rep-5 (O)': sum(rep_results_list_orthogonal)/len(rep_results_list_orthogonal),
                'loc-5 (O)': sum(loc_results_list_orthogonal)/len(loc_results_list_orthogonal),
                'num_lines': sum(num_lines)/len(num_lines)
            })

    logger.info('overall',extra={
                'rep-5 (S)': sum(rep_results_list_structural)/len(rep_results_list_structural),
                'loc-5 (S)': sum(loc_results_list_structural)/len(loc_results_list_structural),
                'rep-5 (O)': sum(rep_results_list_orthogonal)/len(rep_results_list_orthogonal),
                'loc-5 (O)': sum(loc_results_list_orthogonal)/len(loc_results_list_orthogonal),
                'num_lines': sum(num_lines)/len(num_lines)
            })
    logger.info(args)
