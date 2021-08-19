from contextlib import contextmanager
import logging
from os import stat
from typing import List
import numpy as np

from .. import show


try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
    plt = None
    make_axes_locatable = None

LOG = logging.getLogger(__name__)


class Base:
    _image = None
    common_ax = None

    def __init__(self):
        self._ax = None
    
    @staticmethod
    def image(image):
        if image is None:
            Base._image = None
            return
        Base._image = np.asarray(image)
    
    @staticmethod
    def reset():
        Base._image = None

    @contextmanager
    def image_canvas(self, image, *args, **kwargs):
        ax = self._ax or self.common_ax
        if ax is not None:
            ax.set_axis_off()
            ax.imshow(np.asarray(image))
            yield ax
            return
        
        with show.image_canvas(image, *args, **kwargs) as ax:
            yield ax

    @contextmanager
    def canvas(self, *args, **kwargs):
        ax = self._ax or self.common_ax
        if ax is not None:
            yield ax
            return

        with show.canvas(*args, **kwargs) as ax:
            yield ax