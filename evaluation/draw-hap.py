import matplotlib.pyplot as plt
import argparse
import numpy as np
import json
import matplotlib as mpl
import scipy.io as sio
from scipy import interpolate

mpl.rcParams.update({"font.size": 18})
# mpl.font_manager._rebuild()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',nargs='+', type=str)
    parser.add_argument('--threshold', type=int, choices = [5,10,15], default=10)
    parser.add_argument('--dest', default = None, type=str, help='the destination of the saved figure')

    args = parser.parse_args()

    evaluation_results = []
    legends = []
    for path in args.path:
        result = sio.loadmat(path)
        evaluation_results.append(result)
        legends.append(result['label'])
    f_scores = np.linspace(0.2,0.9,num=8).tolist()
    for f_score in f_scores:
        x = np.linspace(0.01,1)
        y = f_score*x/(2*x-f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color=[0,0.5,0], alpha=0.3)
        plt.annotate("f={0:0.1}".format(f_score), xy=(0.9, y[45] + 0.02), alpha=0.4,fontsize=10)

    plt.rc('legend',fontsize=14)
    plt.grid(True)
    plt.axis([0.0, 1.0, 0.0, 1.0])
    plt.xticks(np.arange(0, 1.0, step=0.1))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.yticks(np.arange(0, 1.0, step=0.1))
    
    for label, result in zip(legends,evaluation_results):
        label = label.item()
        precision = result['precision'][0].flatten()
        recall = result['recall'][0].flatten()
        idx = np.isfinite(precision)*np.isfinite(recall)
        precision = precision[idx]
        recall = recall[idx]

        x = np.arange(0,1,0.01)*recall[-1]
        f = interpolate.interp1d(recall,precision,kind='cubic',bounds_error=False,fill_value=precision[0])
        y = f(x)
        T = 0.005

        print(result['f'].item())
        

        #sap_head = "sAP$^{%d}$"%(args.threshold)
        if 'HAWP' in label:
           label_ = '[F={:.1f}] {}(Ours)'.format(result['f'].item()*100,label)
        else:
           label_ = '[F={:.1f}] {}'.format(result['f'].item()*100,label)
        #plt.plot(recall,precision,'-',label=label_sap,linewidth=3)
        # plt.plot(recall,precision,'-')
        plt.plot(x,y,'-',label=label_,linewidth=3)
    plt.legend(loc='best')
    title = "PR Curve for AP$H$"
    plt.title(title)
    if args.dest:
        plt.savefig(args.dest, bbox_inches='tight')
    else:
        plt.show()

if __name__ == "__main__":
    main()