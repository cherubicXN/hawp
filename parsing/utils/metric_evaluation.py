import numpy as np

def msTPFP(line_pred, line_gt, threshold):
    line_pred = line_pred.reshape(-1, 2, 2)[:, :, ::-1]
    line_gt = line_gt.reshape(-1, 2, 2)[:, :, ::-1]
    diff = ((line_pred[:, None, :, None] - line_gt[:, None]) ** 2).sum(-1)
    diff = np.minimum(
        diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
    )

    choice = np.argmin(diff, 1)
    dist = np.min(diff, 1)
    hit = np.zeros(len(line_gt), np.bool)
    tp = np.zeros(len(line_pred), np.float)
    fp = np.zeros(len(line_pred), np.float)
    for i in range(len(line_pred)):
        if dist[i] < threshold and not hit[choice[i]]:
            hit[choice[i]] = True
            tp[i] = 1
        else:
            fp[i] = 1
    return tp, fp


def TPFP(lines_dt, lines_gt, threshold):
    lines_dt = lines_dt.reshape(-1,2,2)[:,:,::-1]
    lines_gt = lines_gt.reshape(-1,2,2)[:,:,::-1]
    diff = ((lines_dt[:, None, :, None] - lines_gt[:, None]) ** 2).sum(-1)
    diff = np.minimum(
        diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
    )

    # diff1 = ((lines_dt[:, None, :2] - lines_gt[:, :2]) ** 2).sum(-1)
    # diff2 = ((lines_dt[:, None, 2:] - lines_gt[:, 2:]) ** 2).sum(-1)
    # diff3 = ((lines_dt[:, None, :2] - lines_gt[:, 2:]) ** 2).sum(-1)
    # diff4 = ((lines_dt[:, None, 2:] - lines_gt[:, :2]) ** 2).sum(-1)
    # import pdb
    # pdb.set_trace()
    # diff = np.minimum(diff1+diff2, diff3+diff4)
    choice = np.argmin(diff,1)
    dist = np.min(diff,1)
    hit = np.zeros(len(lines_gt), np.bool)
    tp = np.zeros(len(lines_dt), np.float)
    fp = np.zeros(len(lines_dt),np.float)

    for i in range(lines_dt.shape[0]):
        if dist[i] < threshold and not hit[choice[i]]:
            hit[choice[i]] = True
            tp[i] = 1
        else:
            fp[i] = 1
    return tp, fp

def AP(tp, fp):
    recall = tp
    precision = tp/np.maximum(tp+fp, 1e-9)

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))



    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    i = np.where(recall[1:] != recall[:-1])[0]

    ap = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])

    return ap
