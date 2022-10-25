import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def non_maximum_suppression(a):
    ap = F.max_pool2d(a, 3, stride=1, padding=1)
    mask = (a == ap).float().clamp(min=0.0)
    
    return a * mask

def get_junctions(jloc, joff, topk = 300, th=0):
    height, width = jloc.size(1), jloc.size(2)
    jloc = jloc.reshape(-1)
    joff = joff.reshape(2, -1)

    
    scores, index = torch.topk(jloc, k=topk)
    # y = (index // width).float() + torch.gather(joff[1], 0, index) + 0.5
    y = torch.div(index,width,rounding_mode='trunc').float()+ torch.gather(joff[1], 0, index) + 0.5
    x = (index % width).float() + torch.gather(joff[0], 0, index) + 0.5

    junctions = torch.stack((x, y)).t()

    return junctions[scores>th], scores[scores>th]

def plot_lines(lines, scale=1.0, color = 'red', **kwargs):
    if isinstance(lines, np.ndarray):
        plt.plot([lines[:,0]*scale,lines[:,2]*scale],[lines[:,1]*scale,lines[:,3]*scale],color=color,linestyle='-')
    else:
        lines_np = lines.detach().cpu().numpy()
        plt.plot([lines_np[:,0]*scale,lines_np[:,2]*scale],[lines_np[:,1]*scale,lines_np[:,3]*scale],color=color,linestyle='-')

    