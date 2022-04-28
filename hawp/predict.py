import argparse
import glob
import os
from tqdm import tqdm

from . import predicting
from . import show
from . import visualizer
import json

def cli():
    parser = argparse.ArgumentParser(
        prog='python -m hawp.predict',
        usage='%(prog)s [options] images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    show.cli(parser)

    parser.add_argument('images', nargs='*',
                        help='input images')
    parser.add_argument('--glob',
                        help='glob expression for input images (for many images)')
    parser.add_argument('-o', '--image-output', default=None, nargs='?', const=True,help='Whether to output an image, '
                             'with the option to specify the output path or directory')
    
    parser.add_argument('--json-output', default=None, nargs='?', const=True,
                        help='Whether to output a json file, '
                             'with the option to specify the output path or directory')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    args = parser.parse_args()

    if args.glob:
        args.images += glob.glob(args.glob)
    if not args.images:
        raise Exception("no image files given")

    show.configure(args)
    return args


def out_name(arg, in_name, default_extension):
    """Determine an output name from args, input name and extension.

    arg can be:
    - none: return none (e.g. show image but don't store it)
    - True: activate this output and determine a default name
    - string:
        - not a directory: use this as the output file name
        - is a directory: use directory name and input name to form an output
    """
    if arg is None:
        return None

    if arg is True:
        return in_name + default_extension

    if os.path.isdir(arg):
        return os.path.join(
            arg,
            os.path.basename(in_name)
        ) + default_extension

    return arg

if __name__ == "__main__":

    args = cli()

    if args.disable_cuda:
        predicting.WireframeParser.device = 'cpu'
    wireframe_parser = predicting.WireframeParser(visualize_image = True)
    
    wireframe_painter = show.painters.WireframePainter()
    
    progress_bar = tqdm(total=len(args.images))
    for predict, _, meta in wireframe_parser.images(args.images):
        if predict is None:
            continue

        if args.json_output is not None:
            predict_json = predict.jsonize()
            json_out_name = out_name(
                args.json_output, meta['filename'], '.wireframe.json',
            )
            with open(json_out_name,'w') as writer:
                json.dump(predict_json,writer)

        if args.show or args.image_output is not None:
            image = visualizer.Base._image
            ext = show.Canvas.out_file_extension
            image_out_name = out_name(args.image_output, meta['filename'], '.wireframe.'+ext)
            with show.image_canvas(image, image_out_name) as ax:
                wireframe_painter.draw_wireframe(ax,predict)

        progress_bar.update(1)
    
    progress_bar.close()
