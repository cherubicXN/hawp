import matplotlib.pyplot as plt
import argparse
import numpy as np
import json
import matplotlib as mpl

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
        with open(path) as f:
            result = json.load(f)
        evaluation_results.append(result['{}'.format(args.threshold)])
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
        precision = result['precision']
        recall = result['recall']
        sap_head = "sAP$^{%d}$"%(args.threshold)
        if 'HAWP' in label:
            label_sap = '[{}={:.1f}] {}(Ours)'.format(sap_head,result['sAP'],label)
        else:
            label_sap = '[{}={:.1f}] {}'.format(sap_head,result['sAP'],label)
        plt.plot(recall,precision,'-',label=label_sap,linewidth=3)
    plt.legend(loc='best')
    title = "PR Curve for sAP$^{%d}$"%(int(args.threshold))
    plt.title(title)
    if args.dest:
        plt.savefig(args.dest, bbox_inches='tight')
    else:
        plt.show()

if __name__ == "__main__":
    main()