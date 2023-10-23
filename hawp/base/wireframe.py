import copy
import math
import numpy as np
import torch
import json

class WireframeGraph:
    def __init__(self, 
                vertices: torch.Tensor, 
                v_confidences: torch.Tensor,
                edges: torch.Tensor, 
                edge_weights: torch.Tensor, 
                frame_width: int, 
                frame_height: int):
        self.vertices = vertices
        self.v_confidences = v_confidences
        self.edges = edges
        self.weights = edge_weights
        self.frame_width = frame_width
        self.frame_height = frame_height

    @classmethod
    def xyxy2indices(cls,junctions, lines):
        # junctions: (N,2)
        # lines: (M,4)
        # return: (M,2)
        dist1 = torch.norm(junctions[None,:,:]-lines[:,None,:2],dim=-1)
        dist2 = torch.norm(junctions[None,:,:]-lines[:,None,2:],dim=-1)
        idx1 = torch.argmin(dist1,dim=-1)
        idx2 = torch.argmin(dist2,dim=-1)
        return torch.stack((idx1,idx2),dim=-1)
    @classmethod
    def load_json(cls, fname):
        with open(fname,'r') as f:
            data = json.load(f)

         
        vertices = torch.tensor(data['vertices'])
        v_confidences = torch.tensor(data['vertices-score'])
        edges = torch.tensor(data['edges'])
        edge_weights = torch.tensor(data['edges-weights'])
        height = data['height']
        width = data['width']

        return WireframeGraph(vertices,v_confidences,edges,edge_weights,width,height)

    @property
    def is_empty(self):
        for key, val in self.__dict__.items():
            if val is None:
                return True
        return False

    @property
    def num_vertices(self):
        if self.is_empty:
            return 0
        return self.vertices.shape[0]
    
    @property
    def num_edges(self):
        if self.is_empty:
            return 0
        return self.edges.shape[0]


    def line_segments(self, threshold = 0.05, device=None, to_np=False):
        is_valid = self.weights>threshold
        p1 = self.vertices[self.edges[is_valid,0]]
        p2 = self.vertices[self.edges[is_valid,1]]
        ps = self.weights[is_valid]

        lines = torch.cat((p1,p2,ps[:,None]),dim=-1)
        if device is not None:
            lines = lines.to(device)
        if to_np:
            lines = lines.cpu().numpy()

        return lines
       # if device != self.device:
        
    def rescale(self, image_width, image_height):
        scale_x = float(image_width)/float(self.frame_width)
        scale_y = float(image_height)/float(self.frame_height)

        self.vertices[:,0] *= scale_x
        self.vertices[:,1] *= scale_y
        self.frame_width = image_width
        self.frame_height = image_height

    def jsonize(self):
        return {
            'vertices': self.vertices.cpu().tolist(),
            'vertices-score': self.v_confidences.cpu().tolist(),
            'edges': self.edges.cpu().tolist(),
            'edges-weights': self.weights.cpu().tolist(),
            'height': self.frame_height,
            'width': self.frame_width,
        }
    def __repr__(self) -> str:
        return "WireframeGraph\n"+\
               "Vertices: {}\n".format(self.num_vertices)+\
               "Edges: {}\n".format(self.num_edges,) + \
               "Frame size (HxW): {}x{}".format(self.frame_height,self.frame_width)

#graph = WireframeGraph()
if __name__ == "__main__":
    graph = WireframeGraph.load_json('NeuS/public_data/bmvs_clock/hawp/000.json')
    print(graph)
    