import torch
import numpy as np
from torch.utils.data.dataloader import default_collate

from hawp.base import _C

class HAFMencoder(object):
    def __init__(self, cfg):
        self.dis_th = cfg.ENCODER.DIS_TH
        self.ang_th = cfg.ENCODER.ANG_TH
        self.num_static_pos_lines = cfg.ENCODER.NUM_STATIC_POS_LINES
        self.num_static_neg_lines = cfg.ENCODER.NUM_STATIC_NEG_LINES
        self.bck_weight = cfg.ENCODER.BACKGROUND_WEIGHT
    def __call__(self,annotations):
        targets = []
        metas   = []
        for ann in annotations:
            t,m = self._process_per_image(ann)
            targets.append(t)
            metas.append(m)
        
        return default_collate(targets),metas

    def adjacent_matrix(self, n, edges, device):
        mat = torch.zeros(n+1,n+1,dtype=torch.bool,device=device)
        if edges.size(0)>0:
            mat[edges[:,0], edges[:,1]] = 1
            mat[edges[:,1], edges[:,0]] = 1
        return mat

    def _process_per_image(self,ann):
        junctions = ann['junctions']
        device = junctions.device
        height, width = ann['height'], ann['width']
        # jmap = torch.zeros((height,width),device=device)
        # joff = torch.zeros((2,height,width),device=device,dtype=torch.float32)
        jmap = np.zeros((height,width),dtype=np.float32)
        joff = np.zeros((2,height,width),dtype=np.float32)
        # junctions[:,0] = junctions[:,0].clamp(min=0,max=width-1)
        # junctions[:,1] = junctions[:,1].clamp(min=0,max=height-1)
        junctions_np = junctions.cpu().numpy()
        xint, yint = junctions_np[:,0].astype(np.long), junctions_np[:,1].astype(np.long)
        off_x = junctions_np[:,0] - np.floor(junctions_np[:,0]) - 0.5
        off_y = junctions_np[:,1] - np.floor(junctions_np[:,1]) - 0.5
        jmap[yint, xint] = 1
        joff[0,yint, xint] = off_x
        joff[1,yint, xint] = off_y
        # xint,yint = junctions[:,0].long(), junctions[:,1].long()
        # off_x = junctions[:,0] - xint.float()-0.5
        # off_y = junctions[:,1] - yint.float()-0.5

        # jmap[yint,xint] = 1
        # joff[0,yint,xint] = off_x
        # joff[1,yint,xint] = off_y
        jmap = torch.from_numpy(jmap).to(device)
        joff = torch.from_numpy(joff).to(device)

        edges_positive = ann['edges_positive']
        edges_negative = ann['edges_negative']
        
        pos_mat = self.adjacent_matrix(junctions.size(0),edges_positive,device)
        neg_mat = self.adjacent_matrix(junctions.size(0),edges_negative,device)        
        lines = torch.cat((junctions[edges_positive[:,0]], junctions[edges_positive[:,1]]),dim=-1)
        lines_neg = torch.cat((junctions[edges_negative[:2000,0]],junctions[edges_negative[:2000,1]]),dim=-1)
        lmap, _, _ = _C.encodels(lines,height,width,height,width,lines.size(0))

        center_points = (lines[:,:2] + lines[:,2:])/2.0
        cmap = torch.zeros((height,width),device=device)
        xx, yy =torch.meshgrid(torch.arange(width,dtype=torch.float32,device=device),torch.arange(height,dtype=torch.float32,device=device),indexing='xy')

        ctl_dis = torch.min((xx[...,None]-center_points[None,None,:,0])**2 + (yy[...,None]-center_points[None,None,:,1])**2,dim=-1)[0]
        cmask = ctl_dis<=4.0

        cxint, cyint = center_points[:,0].long(), center_points[:,1].long()
        cmap[cyint,cxint] = 1


        lpos = np.random.permutation(lines.cpu().numpy())[:self.num_static_pos_lines]
        lneg = np.random.permutation(lines_neg.cpu().numpy())[:self.num_static_neg_lines]
        # lpos = lines[torch.randperm(lines.size(0),device=device)][:self.num_static_pos_lines]
        # lneg = lines_neg[torch.randperm(lines_neg.size(0),device=device)][:self.num_static_neg_lines]
        lpos = torch.from_numpy(lpos).to(device)
        lneg = torch.from_numpy(lneg).to(device)
        
        lpre = torch.cat((lpos,lneg),dim=0)
        _swap = (torch.rand(lpre.size(0))>0.5).to(device)
        lpre[_swap] = lpre[_swap][:,[2,3,0,1]]
        lpre_label = torch.cat(
            [
                torch.ones(lpos.size(0),device=device),
                torch.zeros(lneg.size(0),device=device)
             ])

        meta = {
            'junc': junctions,
            'Lpos':   pos_mat,
            'Lneg':   neg_mat,
            'lpre':      lpre,
            'lpre_label': lpre_label,
            'lines':     lines,
        }


        dismap = torch.sqrt(lmap[0]**2+lmap[1]**2)[None]
        def _normalize(inp):
            mag = torch.sqrt(inp[0]*inp[0]+inp[1]*inp[1])
            return inp/(mag+1e-6)

        md_map = _normalize(lmap[:2])
        st_map = _normalize(lmap[2:4])
        ed_map = _normalize(lmap[4:])

        md_ = md_map.reshape(2,-1).t()
        st_ = st_map.reshape(2,-1).t()
        ed_ = ed_map.reshape(2,-1).t()
        Rt = torch.cat(
                (torch.cat((md_[:,None,None,0],md_[:,None,None,1]),dim=2),
                 torch.cat((-md_[:,None,None,1], md_[:,None,None,0]),dim=2)),dim=1)
        R = torch.cat(
                (torch.cat((md_[:,None,None,0], -md_[:,None,None,1]),dim=2),
                 torch.cat((md_[:,None,None,1], md_[:,None,None,0]),dim=2)),dim=1)

        Rtst_ = torch.matmul(Rt, st_[:,:,None]).squeeze(-1).t()
        Rted_ = torch.matmul(Rt, ed_[:,:,None]).squeeze(-1).t()
        swap_mask = (Rtst_[1]<0)*(Rted_[1]>0)
        pos_ = Rtst_.clone()
        neg_ = Rted_.clone()
        temp = pos_[:,swap_mask]
        pos_[:,swap_mask] = neg_[:,swap_mask]
        neg_[:,swap_mask] = temp

        pos_[0] = pos_[0]#.clamp(min=1e-9)
        pos_[1] = pos_[1]#.clamp(min=1e-9)
        neg_[0] = neg_[0]#.clamp(min=1e-9)
        neg_[1] = neg_[1]#.clamp(max=-1e-9)
        
        mask = (dismap.view(-1)<=self.dis_th).float()

        pos_map = pos_.reshape(-1,height,width)
        neg_map = neg_.reshape(-1,height,width)

        md_angle  = torch.atan2(md_map[1], md_map[0])
        pos_angle = torch.atan2(pos_map[1],pos_map[0])
        neg_angle = torch.atan2(neg_map[1],neg_map[0])
        mask *= (pos_angle.reshape(-1)>self.ang_th*np.pi/2.0)
        mask *= (neg_angle.reshape(-1)<-self.ang_th*np.pi/2.0)

        pos_angle_n = pos_angle/(np.pi/2)
        neg_angle_n = -neg_angle/(np.pi/2)
        md_angle_n  = md_angle/(np.pi*2) + 0.5
        mask    = mask.reshape(height,width)

        mask[mask<1e-3] = self.bck_weight
        # import pdb; pdb.set_trace()
        hafm_ang = torch.cat((md_angle_n[None],pos_angle_n[None],neg_angle_n[None],),dim=0)
        hafm_dis   = dismap.clamp(max=self.dis_th)/self.dis_th
        mask = mask[None]
        target = {'jloc':jmap[None],
                'joff':joff,
                'cloc': cmap[None],
                'md': hafm_ang,
                'dis': hafm_dis,
                'mask': mask
               }
        return target, meta