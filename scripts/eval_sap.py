from parsing.config.paths_catalog import DatasetCatalog
from parsing.utils.metric_evaluation import TPFP, AP
import argparse
import os
import os.path as osp
from termcolor import colored
import numpy as np
import json
import matplotlib.pyplot as plt

AVAILABLE_DATASETS = ('wireframe_test', 'york_test')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser('Structural AP Evaluation')
    argparser.add_argument('--path',dest='path',type=str,required=True)
    argparser.add_argument('-t','--threshold', dest='threshold', type=float, default=10.0)

    args = argparser.parse_args()

    result_path = args.path

    assert result_path.endswith('.json'), \
        'The result file has to end with .json'

    dataset_name = osp.basename(result_path).rstrip('.json')
    
    assert dataset_name in AVAILABLE_DATASETS, \
        'Currently, we only support  {} datasets for evaluation'.format(
            colored(str(AVAILABLE_DATASETS),'red')
        )
    ann_file = DatasetCatalog.get(dataset_name)['args']['ann_file']
    
    with open(ann_file,'r') as _ann:
        annotations_list = json.load(_ann)
    
    annotations_dict = {
        ann['filename']: ann for ann in annotations_list
    }

    with open(result_path,'r') as _res:
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
    print(sAP)
    try:
        plt.plot(rcs,pcs,'r-')
        plt.title(str(sAP))
        plt.show()
    except:
        pass
    # import pdb; pdb.set_trace()