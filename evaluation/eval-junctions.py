import json
import numpy as np
import argparse
from hawp.utils.metric_evaluation import TPFP, AP
from tqdm import tqdm 
DIST = [0.5, 1.0, 2.0]

BENCHMARKS = {
    'wireframe': 'data/wireframe/test.json',
    'york': 'data/york/test.json',
}


def APJ(vert_pred, vert_gt, max_distance, im_ids):
    if len(vert_pred) == 0:
        return 0

    vert_pred = np.array(vert_pred)
    vert_gt = np.array(vert_gt)

    confidence = vert_pred[:, -1]
    idx = np.argsort(-confidence)
    vert_pred = vert_pred[idx, :]
    im_ids = im_ids[idx]
    n_gt = sum(len(gt) for gt in vert_gt)

    nd = len(im_ids)
    tp, fp = np.zeros(nd, dtype=np.float32), np.zeros(nd, dtype=np.float32)
    hit = [[False for _ in j] for j in vert_gt]

    for i in range(nd):
        gt_juns = vert_gt[im_ids[i]]
        pred_juns = vert_pred[i][:-1]
        if len(gt_juns) == 0:
            continue
        dists = np.linalg.norm((pred_juns[None, :] - gt_juns), axis=1)
        choice = np.argmin(dists)
        dist = np.min(dists)
        if dist < max_distance and not hit[im_ids[i]][choice]:
            tp[i] = 1
            hit[im_ids[i]][choice] = True
        else:
            fp[i] = 1

    tp = np.cumsum(tp) / n_gt
    fp = np.cumsum(fp) / n_gt
    return AP(tp, fp)

def convert_lines_to_junctions(lines,scores, nms_threshold=0):
    junctions = np.concatenate((lines[:,:2],lines[:,2:]))
    scores = np.concatenate((scores,scores))
    idx = np.argsort(-scores)
    junctions = junctions[idx]
    scores = scores[idx]
    # np.unique(junctions,axis=1)
    
    if nms_threshold>0:
        dist = np.sqrt(np.sum((junctions[:,None]-junctions[None])**2,axis=-1))
        num_junctions = junctions.shape[0]
        is_kept = np.ones(num_junctions,dtype=np.bool)        
        for i in range(num_junctions):
            if not is_kept[i]:
                continue
            is_kept[i+1:] = dist[i,i+1:]>nms_threshold
        junctions = junctions[is_kept]
        scores = scores[is_kept]   
    return junctions, scores

def evaluate(annotations_dict, result_list, threshold, increasing_order=False, 
    nms_threshold = 0):
    tp_list, fp_list, scores_list = [],[],[]
    n_gt = 0

    all_junc = np.zeros((0,3))
    all_junc_ids = np.zeros(0,dtype=np.int32)
    all_jc_gt = []

    for i, res in enumerate(tqdm(result_list)):
        filename = res['filename']
        gt = annotations_dict[filename]

        if 'juncs_pred' in res:
            junctions = np.array(res['juncs_pred'],dtype=np.float32)
            scores = np.array(res['juncs_score'],dtype=np.float32)
        else:
            lines_pred = np.array(res['lines_pred'],dtype=np.float32)
            scores = np.array(res['lines_score'],dtype=np.float32)
            if increasing_order:
                scores *= -1

            junctions, scores = convert_lines_to_junctions(lines_pred,scores, nms_threshold=nms_threshold)
        junctions_gt = np.array(gt['junc'],dtype=np.float32)


        junctions[:,0] *= float(128/res['width'])
        junctions[:,1] *= float(128/res['height'])
        junctions = np.concatenate((junctions,scores[:,None]),axis=-1)

        junctions_gt[:,0] *= float(128/gt['width'])
        junctions_gt[:,1] *= float(128/gt['height'])

        all_junc = np.vstack((all_junc,junctions))
        all_jc_gt.append(junctions_gt)

        all_junc_ids = np.hstack((all_junc_ids,np.array([i]*len(junctions))))
    all_junc_ids = all_junc_ids.astype(np.int64)

    apj = sum(APJ(all_junc,all_jc_gt,th,all_junc_ids) for th in DIST)/len(DIST)
    return apj
        # sort_idx = np.argsort(-scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred',type=str,required=True,help='the json file for the wireframe or line segment predictions')
    parser.add_argument('--benchmark', type=str, choices = ['wireframe','york'], required=True)
    parser.add_argument('--increasing-order', default=False, action='store_true')
    parser.add_argument('--nms', type=float, default=0, help='the nms threshold')
    # parser.add_argument('--label',type=str,required=True)

    args = parser.parse_args()
    

    with open(args.pred) as f:
        result_list = json.load(f)
    
    with open(BENCHMARKS[args.benchmark]) as f:
        annotations_list = json.load(f)

    annotations_dict = {
        ann['filename']: ann for ann in annotations_list
    }

    apj = evaluate(annotations_dict,result_list,1.0,args.increasing_order,nms_threshold=args.nms)
    print(apj)

if __name__ == "__main__":
    main()