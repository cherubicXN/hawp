import pdb
import torch
from hawp import show
from hawp.config import cfg
from hawp.utils.comm import to_device
from hawp.dataset.build import build_transform

from hawp.detector import get_hawp_model
from hawp.utils.logger import setup_logger
from skimage import io
import argparse
import matplotlib.pyplot as plt
import numpy as np

from . import dataset
from . import visualizer

class WireframeParser(object):
    loader_workers = None
    device = 'cuda'
    def __init__(self, json_data = False, 
                       visualize_image = False,
                       visualize_processed_image = False):
        self.model = get_hawp_model(pretrained=True).eval()
        self.model = self.model.to(self.device)
        self.preprocessor_transform = build_transform(cfg)
        self.visualize_image = visualize_image
        self.visualize_processed_image = visualize_processed_image

    def dataset(self, data):
        loader_workers = self.loader_workers
        if loader_workers is None:
            loader_workers = 1
        dataloader = torch.utils.data.DataLoader(
            data, batch_size=1, shuffle=False,
            pin_memory=True,
            num_workers = loader_workers,
            collate_fn = dataset.test_dataset.collate_images_anns_meta,
        )
        yield from self.dataloader(dataloader)

    def dataloader(self, dataloader):
        for batch_i, item in enumerate(dataloader):
            if len(item) == 3:
                processed_image_batch, gt_anns_batch, meta_batch = item
                image_batch = [None for _ in processed_image_batch]
            elif len(item) == 4:
                image_batch, processed_image_batch, gt_anns_batch, meta_batch = item
            
            if self.visualize_image:
                visualizer.Base.image(image_batch[0])
            processed_image_batch = processed_image_batch.to(self.device)
            with torch.no_grad():
                wireframe, _ = self.model(processed_image_batch, meta_batch)

            yield wireframe, gt_anns_batch[0], meta_batch[0]

    
    def images(self, file_names, **kwargs):
        data = dataset.test_dataset.ImageList(
            file_names, transform=self.preprocessor_transform, with_raw_image=True)
        yield from self.dataset(data, **kwargs)





def predict(cfg, impath, args):
    wparser = WireframeParser()
    wireframe = wparser(impath)
    transform = build_transform(cfg)
    image = io.imread(impath)
    # wparser(image)
    # image_tensor = transform(image.astype(float))[None].to(device)
    # meta = {
    #     'filename': impath,
    #     'height': image.shape[0],
    #     'width': image.shape[1],
    # }
    
    # with torch.no_grad():
    #     output, _ = model(image_tensor,[meta])
    #     output = to_device(output,'cpu')
    show.Canvas.show = False

    painter = show.painters.WireframePainter()

    with show.image_canvas(image) as ax:
        painter.draw_wireframe(ax,wireframe)
    # import pdb; pdb.set_trace()
        

    # plt.figure(figsize=(6,6))    
    # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
    #         hspace = 0, wspace = 0)
    # plt.imshow(image)
    # plt.plot([lines[idx,0],lines[idx,2]],
    #                     [lines[idx,1],lines[idx,3]], 'b-')
    # plt.plot(lines[idx,0],lines[idx,1],'c.')                        
    # plt.plot(lines[idx,2],lines[idx,3],'c.')                        
    # plt.axis('off')
    # plt.show()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HAWP Testing')

    parser.add_argument("img",type=str,
                        help="image path")                    

    parser.add_argument("--threshold",
                        type=float,
                        default=0.97)

    args = parser.parse_args()

    cfg.freeze()

    predict(cfg,args.img,args)

