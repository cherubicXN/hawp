import os
import cv2
import os.path as osp

SRC = 'epnet/data/york/images'
DES = 'tt/york_test'
if __name__ == '__main__':

    files = os.listdir(SRC)
    os.makedirs(DES,exist_ok=True)
    lst = []
    for f in files:
        base = f.rstrip('.png')
        im = cv2.imread(osp.join(SRC,f))
        lst.append(base+'\n')
        cv2.imwrite(osp.join(DES,base+'_rgb.png'),im)

    with open('tt/york_test.txt','w') as handle:
        handle.writelines(lst)