import logging

import numpy as np
import torch 


try:
    import matplotlib
    import matplotlib.animation
    import matplotlib.collections
    import matplotlib.patches
except ImportError:
    matplotlib = None


LOG = logging.getLogger(__name__)


class HAWPainter:
    line_width = None
    marker_size = None
    confidence_threshold = 0.05

    def __init__(self):

        if self.line_width is None:
            self.line_width = 2
        
        if self.marker_size is None:
            self.marker_size = max(1, int(self.line_width * 0.5))

    def draw_wireframe(self, ax, wireframe, *,
            edge_color = None, vertex_color = None):
        if wireframe is None:
            return
        
        if edge_color is None:
            edge_color = 'b'
        if vertex_color is None:
            vertex_color = 'c'
        
        line_segments = wireframe['lines_pred'][wireframe['lines_score']>self.confidence_threshold]
        
        if isinstance(line_segments, torch.Tensor):
            line_segments = line_segments.cpu().numpy()

        # line_segments = wireframe.line_segments(threshold=self.confidence_threshold)
        # line_segments = line_segments.cpu().numpy()
        ax.plot([line_segments[:,0],line_segments[:,2]],[line_segments[:,1],line_segments[:,3]],'-',color=edge_color)
        ax.plot(line_segments[:,0],line_segments[:,1],'.',color=vertex_color)
        ax.plot(line_segments[:,2],line_segments[:,3],'.',
        color=vertex_color)
