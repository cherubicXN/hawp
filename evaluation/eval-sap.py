import json
import numpy as np
import argparse
from hawp.utils.metric_evaluation import TPFP, AP

BENCHMARKS = {
    'wireframe': 'data/wireframe/test.json',
    'york': 'data/york/test.json',
}

def evaluate(annotations_dict, result_list, threshold, increasing_order=False):
    tp_list, fp_list, scores_list = [],[],[]
    n_gt = 0

    for res in result_list:
        filename = res['filename']
        gt = annotations_dict[filename]
        lines_pred = np.array(res['lines_pred'],dtype=np.float32)
        scores = np.array(res['lines_score'],dtype=np.float32)
        if increasing_order:
            scores *= -1
    
        sort_idx = np.argsort(-scores)
    
        lines_pred[:,0] *= 128/float(res['width'])
        lines_pred[:,1] *= 128/float(res['height'])
        lines_pred[:,2] *= 128/float(res['width'])
        lines_pred[:,3] *= 128/float(res['height'])

        lines_gt   = np.array(gt['lines'],dtype=np.float32)
        lines_gt[:,0]  *= 128/float(gt['width'])
        lines_gt[:,1]  *= 128/float(gt['height'])
        lines_gt[:,2]  *= 128/float(gt['width'])
        lines_gt[:,3]  *= 128/float(gt['height'])
        
        # assert gt['width'] == res['width'] and gt['height'] == res['height']
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
    F_list = (2*rcs*pcs/(rcs+pcs))
    F_list = np.nan_to_num(F_list, 0)
    F = F_list.max()
    
    P = pcs[F_list.argmax()]
    R = rcs[F_list.argmax()]

    sAP = AP(tp,fp)*100

    return {
        'recall': rcs.tolist(),
        'precision': pcs.tolist(),
        'P': P.item(),
        'R': R.item(),
        'F': F.item(),
        'sAP': sAP,
    }
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred',type=str,required=True,help='the json file for the wireframe or line segment predictions')
    parser.add_argument('--benchmark', type=str, choices = ['wireframe','york'], required=True)
    parser.add_argument('--threshold', type=float, choices = [5,10,15])
    parser.add_argument('--increasing-order', default=False, action='store_true')
    # parser.add_argument('--orders', type=str, choices = ['increase', 'decrease'],default = 'decrease', help='the order of the detections by scores. If you choose increase, the line segments with smaller scores will be more confident.')
    parser.add_argument('--label',type=str,required=True)


    args = parser.parse_args()
    

    with open(args.pred) as f:
        result_list = json.load(f)
    
    with open(BENCHMARKS[args.benchmark]) as f:
        annotations_list = json.load(f)

    annotations_dict = {
        ann['filename']: ann for ann in annotations_list
    }

    sAP_reports = {}
    for threshold in [5,10,15]:
        report = evaluate(annotations_dict, result_list, threshold, args.increasing_order)
        print('threshold = {}'.format(threshold))

        for key, val in report.items():
            if isinstance(val, list):
                continue
            print('{}: {}'.format(key,val))
    
        sAP_reports[threshold] = report
    
    sAP_reports['label'] = args.label


    print('saving the evaluation results to: {}'.format(args.pred+'.sap'))
    with open(args.pred+'.sap','w') as f:
        json.dump(sAP_reports,f)

if __name__ == "__main__":
    main()