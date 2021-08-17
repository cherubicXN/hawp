import torch
from parsing.config import cfg
from parsing.utils.comm import to_device
from parsing.dataset import build_test_dataset
from parsing.detector import get_hawp_model
from parsing.utils.logger import setup_logger
from parsing.utils.checkpoint import DetectronCheckpointer
from parsing.config.paths_catalog import DatasetCatalog
from parsing.utils.metric_evaluation import TPFP, AP
import os
import os.path as osp
import argparse
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import numpy as np
AVAILABLE_DATASETS = ('wireframe_test', 'york_test')

parser = argparse.ArgumentParser(description='HAWP Testing')

parser.add_argument("--config-file",
                    metavar="FILE",
                    help="path to config file",
                    type=str,
                    default=None,
                    )

parser.add_argument("--display",
                    default=False,
                    action='store_true')
parser.add_argument('-t','--threshold', dest='threshold', type=float, default=10.0, help="the threshold for sAP evaluation")
parser.add_argument("opts",
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER
                    )
args = parser.parse_args()

def test(cfg):
    logger = logging.getLogger("hawp.testing")
    device = cfg.MODEL.DEVICE
    model = get_hawp_model(pretrained=args.config_file is None)
    model = model.to(device)

    test_datasets = build_test_dataset(cfg)
    
    output_dir = cfg.OUTPUT_DIR
    if args.config_file is not None:
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

        if name not in AVAILABLE_DATASETS:
            continue
        logger.info('evaluating the results on the {} dataset'.format(name))
        ann_file = DatasetCatalog.get(name)['args']['ann_file']
        with open(ann_file,'r') as _ann:
            annotations_list = json.load(_ann)
        annotations_dict = {
            ann['filename']: ann for ann in annotations_list
        }
        with open(outpath_dataset,'r') as _res:
            result_list = json.load(_res)
        
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

            tp, fp = TPFP(lines_pred,lines_gt,args.threshold)
            
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
        sAP_string = 'sAP{} = {:.1f}'.format(args.threshold,sAP)
        logger.info(sAP_string)
        try:
            f_scores = np.linspace(0.2,0.9,num=8)
            for f_score in f_scores:
                x = np.linspace(0.01,1)
                y = f_score*x/(2*x-f_score)
                l, = plt.plot(x[y >= 0], y[y >= 0], color=[0,0.5,0], alpha=0.3)
                plt.annotate("f={0:0.1}".format(f_score), xy=(0.9, y[45] + 0.02), alpha=0.4,fontsize=10)
            
            plt.rc('legend',fontsize=10)
            plt.grid(True)
            plt.axis([0.0, 1.0, 0.0, 1.0])
            plt.xticks(np.arange(0, 1.0, step=0.1))
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.yticks(np.arange(0, 1.0, step=0.1))
            plt.plot(rcs,pcs,'r-')
            plt.title(sAP_string)
            plt.show()
        except:
            pass


if __name__ == "__main__":
    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)
    else:
        cfg.OUTPUT_DIR = 'outputs/default'
        os.makedirs(cfg.OUTPUT_DIR,exist_ok=True)
    
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    
    output_dir = cfg.OUTPUT_DIR
    
    logger = setup_logger('hawp', output_dir)
    logger.info(args)
    if args.config_file is not None:
        logger.info("Loaded configuration file {}".format(args.config_file))
    else:
        logger.info("Loaded the default configuration for testing")

    test(cfg)

