import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse
import os
import os.path as osp
import json
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred',type=str,required=True,help='the json file for the wireframe or line segment predictions')
    parser.add_argument('--benchmark', type=str, choices = ['wireframe','york'], required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--cmp', default='g', choices = ['g','l'])
    parser.add_argument('--dest', type=str, required=True)
    parser.add_argument('--width', default=0, type=int,)
    parser.add_argument('--height', default=0, type=int,)
    parser.add_argument('--topk', default=-1, type=int)
    parser.add_argument('--fname', default=None, type=str)

    args = parser.parse_args()
    with open(args.pred,'r') as fin:
        results = json.load(fin)

    if args.benchmark == 'wireframe':
        args.images = 'data/wireframe/images'
    elif args.benchmark == 'york':
        args.images = 'data/york/images'

    os.makedirs(args.dest,exist_ok=True)

    if args.fname is not None:
        results = [r for r in results if r['filename'] == args.fname]
    for result in results:
        fname = result['filename']

        image = cv2.imread(osp.join(args.images,fname))[:,:,::-1]
        ori_shape = image.shape 

        lines = np.array(result['lines_pred'])
        score = np.array(result['lines_score'])
        if result['width']!=ori_shape[1] or result['height']!=ori_shape[0]:
            sx = float(ori_shape[1]/result['width'])
            sy = float(ori_shape[0]/result['height'])
            sxy = np.array([sx,sy,sx,sy]).reshape(-1,4)
            lines = lines*sxy

        if args.cmp == 'g':
            sort_arg = np.argsort(score)[::-1]
            lines = lines[sort_arg]
            score = score[sort_arg]
            idx = score>args.threshold
        else:
            sort_arg = np.argsort(score)
            lines = lines[sort_arg]
            score = score[sort_arg]
            idx = score<args.threshold
        
        if args.topk > 0:
            idx = np.arange(idx.shape[0])<args.topk
        
        fig = plt.figure()
        fig.set_size_inches(ori_shape[1]/ori_shape[0],1,forward=False)

        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.xlim([-0.5, ori_shape[1]-0.5])
        plt.ylim([ori_shape[0]-0.5, -0.5])

        
        plt.imshow(image)
        plt.plot([lines[idx,0],lines[idx,2]],[lines[idx,1],lines[idx,3]],'-',linewidth=0.5)
        plt.scatter(lines[idx,0],lines[idx,1],color='b',s=1.2,edgecolors='none',zorder=5)
        plt.scatter(lines[idx,2],lines[idx,3],color='b',s=1.2,edgecolors='none',zorder=5)
        plt.axis('off')

        output_path = osp.join(args.dest,fname.replace('png','pdf'))
        print(output_path)
        plt.savefig(output_path,dpi=300,bbox_inches=0)
        plt.close()
        

if __name__ == "__main__":
    main() 