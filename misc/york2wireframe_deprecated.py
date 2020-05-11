import json
import scipy.io as sio
import os
import os.path as osp
import cv2
import matplotlib.pyplot as plt
import numpy as np
OUTPATH = 'epnet/data/york'
SRC = '/home/nxue2/lcnn/data/york_lcnn'
if __name__ == '__main__':

    files = [f for f in os.listdir('YorkUrbanDB') if osp.isdir(osp.join('YorkUrbanDB',f))]

    os.makedirs(OUTPATH,exist_ok=True)
    os.makedirs(osp.join(OUTPATH,'images'),exist_ok=True)

    annotations = []
    for f in files:
        img = cv2.imread(osp.join('YorkUrbanDB',f,f+'.jpg'))
        cv2.imwrite(osp.join(OUTPATH,'images',f+'.png'),img)
        mat = sio.loadmat(osp.join('YorkUrbanDB',f,f+'LinesAndVP.mat'))
        junc = mat['lines']
        lines = np.concatenate((junc[0::2],junc[1::2]),axis=1)
        lines = np.array(lines,dtype=np.float32)
        junc = np.array(np.unique(junc,axis=0),dtype=np.float32)

        print(lines.shape[0], junc.shape[0])
        # import pdb
        # pdb.set_trace()
        annotations.append(
            {
                'filename': f+'.png',
                'height': img.shape[0],
                'width': img.shape[1],
                'lines': lines.tolist(),
                'junc': junc.tolist(),
            }
        )

        plt.imshow(img)
        plt.plot(junc[:,0],junc[:,1],'r.')
        plt.plot([lines[:,0],lines[:,2]],[lines[:,1],lines[:,3]],'r-')
        plt.show()
        # plt.show()
    with open(osp.join(OUTPATH,'test.json'),'w') as buffer:
        json.dump(annotations,buffer)
    # import pdb
    # pdb.set_trace()