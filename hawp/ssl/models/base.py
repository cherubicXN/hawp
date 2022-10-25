import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

class HAWPBase(nn.Module):
    def __init__(self, *, 
        num_points,
        num_residuals,
        distance_threshold,
        ):
        super(HAWPBase,self).__init__()

        self.num_points = num_points
        self.num_residuals = num_residuals
        self.distance_threshold = distance_threshold
        
        self.register_buffer('tspan', torch.linspace(0, 1, self.num_points)[None,None,:])

    @staticmethod
    def bilinear_sampling(features, points):
        h,w = features.size(1), features.size(2)
        px, py = points[:,0], points[:,1]

        px0 = px.floor().clamp(min=0, max=w-1)
        py0 = py.floor().clamp(min=0, max=h-1)
        px1 = (px0 + 1).clamp(min=0, max=w-1)
        py1 = (py0 + 1).clamp(min=0, max=h-1)
        px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()
        # import pdb; pdb.set_trace()
        xp = features[:, py0l, px0l] * (py1-py) * (px1 - px)+ features[:, py1l, px0l] * (py - py0) * (px1 - px)+ features[:, py0l, px1l] * (py1 - py) * (px - px0)+ features[:, py1l, px1l] * (py - py0) * (px - px0)

        return xp

    @staticmethod
    def get_line_points(lines_per_im):
        U,V = lines_per_im[:,:2], lines_per_im[:,2:]
        sampled_points = U[:,:,None]*self.tspan + V[:,:,None]*(1-self.tspan) -0.5
        return sampled_points

    @staticmethod
    def compute_loi_features(features_per_image, lines_per_im, tspan):
        num_channels = features_per_image.shape[0]
        h,w = features_per_image.size(1), features_per_image.size(2)
        U,V = lines_per_im[:,:2], lines_per_im[:,2:]
        
        sampled_points = U[:,:,None]*tspan + V[:,:,None]*(1-tspan) -0.5

        sampled_points = sampled_points.permute((0,2,1)).reshape(-1,2)
        px,py = sampled_points[:,0],sampled_points[:,1]
        px0 = px.floor().clamp(min=0, max=w-1)
        py0 = py.floor().clamp(min=0, max=h-1)
        px1 = (px0 + 1).clamp(min=0, max=w-1)
        py1 = (py0 + 1).clamp(min=0, max=h-1)
        px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()
        #px0l = px0l.clamp(min=0, max=w-1)
        #py0l = px0l.clamp(min=0, max=h-1)
        #px1l = px0l.clamp(min=0, max=w-1)
        #py1l = px0l.clamp(min=0, max=h-1)
        
        xp = features_per_image[:, py0l, px0l] * (py1-py) * (px1 - px)+ features_per_image[:, py1l, px0l] * (py - py0) * (px1 - px)+ features_per_image[:, py0l, px1l] * (py1 - py) * (px - px0)+ features_per_image[:, py1l, px1l] * (py - py0) * (px - px0)
        xp = xp.reshape(features_per_image.shape[0],-1,tspan.numel()).permute(1,0,2).contiguous()

        return xp.flatten(1)

    def hafm_decoding(self,md_maps, dis_maps, residual_maps, scale=5.0, flatten = True):

        device = md_maps.device
        scale = self.distance_threshold

        batch_size, _, height, width = md_maps.shape
        _y = torch.arange(0,height,device=device).float()
        _x = torch.arange(0,width, device=device).float()

        y0, x0 =torch.meshgrid(_y, _x,indexing='ij')
        y0 = y0[None,None]
        x0 = x0[None,None]
        
        sign_pad = torch.arange(-self.num_residuals,self.num_residuals+1,device=device,dtype=torch.float32).reshape(1,-1,1,1)

        if residual_maps is not None:
            residual = residual_maps*sign_pad
            distance_fields = dis_maps + residual
        else:
            distance_fields = dis_maps
        distance_fields = distance_fields.clamp(min=0,max=1.0)
        md_un = (md_maps[:,:1] - 0.5)*np.pi*2
        st_un = md_maps[:,1:2]*np.pi/2.0
        ed_un = -md_maps[:,2:3]*np.pi/2.0

        cs_md = md_un.cos()
        ss_md = md_un.sin()

        y_st = torch.tan(st_un)
        y_ed = torch.tan(ed_un)

        x_st_rotated = (cs_md - ss_md*y_st)*distance_fields*scale
        y_st_rotated = (ss_md + cs_md*y_st)*distance_fields*scale

        x_ed_rotated = (cs_md - ss_md*y_ed)*distance_fields*scale
        y_ed_rotated = (ss_md + cs_md*y_ed)*distance_fields*scale

        x_st_final = (x_st_rotated + x0).clamp(min=0,max=width-1)
        y_st_final = (y_st_rotated + y0).clamp(min=0,max=height-1)

        x_ed_final = (x_ed_rotated + x0).clamp(min=0,max=width-1)
        y_ed_final = (y_ed_rotated + y0).clamp(min=0,max=height-1)

        
        lines = torch.stack((x_st_final,y_st_final,x_ed_final,y_ed_final),dim=-1)
        if flatten:
            lines = lines.reshape(batch_size,-1,4)

        return lines
    
    @staticmethod
    def non_maximum_suppression(a):
        ap = F.max_pool2d(a, 3, stride=1, padding=1)
        mask = (a == ap).float().clamp(min=0.0)
        
        return a * mask

    @staticmethod
    def get_junctions(jloc, joff, topk = 300, th=0):
        height, width = jloc.size(1), jloc.size(2)
        jloc = jloc.reshape(-1)
        joff = joff.reshape(2, -1)

        
        scores, index = torch.topk(jloc, k=topk)
        # y = (index // width).float() + torch.gather(joff[1], 0, index) + 0.5
        y = torch.div(index,width,rounding_mode='trunc').float()+ torch.gather(joff[1], 0, index) + 0.5
        x = (index % width).float() + torch.gather(joff[0], 0, index) + 0.5

        junctions = torch.stack((x, y)).t()
        
        if th>0 :
            return junctions[scores>th], scores[scores>th]
        else:
            return junctions, scores

if __name__ == "__main__":
    base = HAWPBase()
    import pdb; pdb.set_trace()