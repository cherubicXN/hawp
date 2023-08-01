# from hawp.config import defaults
import logging

from .canvas import Canvas
from .painters import HAWPainter
import matplotlib
LOG = logging.getLogger(__name__)

def cli(parser):
    group = parser.add_argument_group('show')

    assert not Canvas.show
    group.add_argument('--show', default=False,action='store_true',
                help='show every plot, i.e., call matplotlib show()')

    group.add_argument('--edge-threshold', default=None, type=float,
                help='show the wireframe edges whose confidences are greater than [edge_threshold]')
    group.add_argument('--out-ext', default='png', type=str,
                help='save the plot in specific format')
def configure(args):
    Canvas.show = args.show
    Canvas.out_file_extension = args.out_ext
    if args.edge_threshold is not None:
        HAWPainter.confidence_threshold = args.edge_threshold