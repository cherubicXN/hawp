import torch
from parsing.config import cfg
from parsing.utils.comm import to_device
from parsing.dataset.build import build_transform
from parsing.detector import get_hawp_model
from parsing.utils.logger import setup_logger
from skimage import io
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='HAWP Testing')

parser.add_argument("--img",type=str,required=True,
                    help="image path")                    

parser.add_argument("--threshold",
                    type=float,
                    default=0.97)

args = parser.parse_args()


def test(cfg, impath):
    device = cfg.MODEL.DEVICE
    model = get_hawp_model(pretrained=True)
    model = model.to(device)

    transform = build_transform(cfg)
    image = io.imread(impath)
    image_tensor = transform(image.astype(float))[None].to(device)
    meta = {
        'filename': impath,
        'height': image.shape[0],
        'width': image.shape[1],
    }
    
    with torch.no_grad():
        output, _ = model(image_tensor,[meta])
        output = to_device(output,'cpu')

    
    lines = output['lines_pred'].numpy()
    scores = output['lines_score'].numpy()
    idx = scores>args.threshold

    plt.figure(figsize=(6,6))    
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
    plt.imshow(image)
    plt.plot([lines[idx,0],lines[idx,2]],
                        [lines[idx,1],lines[idx,3]], 'b-')
    plt.plot(lines[idx,0],lines[idx,1],'c.')                        
    plt.plot(lines[idx,2],lines[idx,3],'c.')                        
    plt.axis('off')
    plt.show()
    
if __name__ == "__main__":
    cfg.freeze()
    test(cfg,args.img)

