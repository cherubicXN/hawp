import numpy as np
import scipy.io as sio
import os.path as osp
import glob

def ap(tp, fp):
    recall = tp
    precision = tp / np.maximum(tp + fp, 1e-9)

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    i = np.where(recall[1:] != recall[:-1])[0]
    return np.sum((recall[i + 1] - recall[i]) * precision[i + 1])

def msTPFP(lines_pred, lines_gt, threshold = 5):
    x1_pred = lines_pred[:,:2]
    x2_pred = lines_pred[:,2:4]
    x1_gt = lines_gt[:,:2]
    x2_gt = lines_gt[:,2:]
    diff1_1 = ((x1_pred[:,None]-x1_gt)**2).sum(-1)
    diff1_2 = ((x1_pred[:,None]-x2_gt)**2).sum(-1)

    diff2_1 = ((x2_pred[:, None] - x1_gt) ** 2).sum(-1)
    diff2_2 = ((x2_pred[:, None] - x2_gt) ** 2).sum(-1)

    diff = np.minimum(diff1_1+diff2_2, diff1_2+diff2_1)

    choice = np.argmin(diff,1)
    dist = np.min(diff,1)
    hit = np.zeros(len(lines_gt),np.bool)
    tp = np.zeros(len(lines_pred),np.float)
    fp = np.zeros(len(lines_pred), np.float)
    for i in range(len(lines_pred)):
        if dist[i] < threshold and not hit[choice[i]]:
            hit[choice[i]] = True
            tp[i] = 1
        else:
            fp[i] = 1

    return tp,fp


if __name__ == '__main__':

    path = osp.join('outputs','afmbox_R50-FPN-AFM-512','wireframe','*.mat')

    files = glob.glob(path)

    tps, fps, scores = [],[],[]
    n_gt = 0

    aps = []
    for f in files:
        mat = sio.loadmat(f)

        height = mat['height'].item()
        width  = mat['width'].item()
        lines_pred = mat['pred']
        lines_gt   = mat['gt']

        lines_pred[:, 0] *= 128 / width
        lines_pred[:, 2] *= 128 / width
        lines_pred[:, 1] *= 128 / height
        lines_pred[:, 3] *= 128 / height

        lines_gt[:, 0] *= 128 / width
        lines_gt[:, 2] *= 128 / width
        lines_gt[:, 1] *= 128 / height
        lines_gt[:, 3] *= 128 / height

        pred_score = lines_pred[:,4]


        n_gt += len(lines_gt)

        tp,fp = msTPFP(lines_pred,lines_gt,10)
        # import pdb
        # pdb.set_trace()
        # tp = np.cumsum(tp)/len(lines_gt)
        # fp = np.cumsum(fp)/len(lines_gt)

        aps += [ap(tp,fp)]
        tps.append(tp)
        fps.append(fp)
        scores.append(pred_score)

    tps = np.concatenate(tps)
    fps = np.concatenate(fps)
    scores = np.concatenate(scores)
    index = np.argsort(scores)
    tp = np.cumsum(tps[index])/n_gt
    fp = np.cumsum(fps[index])/n_gt
    import pdb
    pdb.set_trace()


