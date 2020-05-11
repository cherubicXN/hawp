import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import  numpy as np
import time

from parsing.encoder.hafm import HAFMencoder
from parsing.backbones import build_backbone

def cross_entropy_loss_for_junction(logits, positive):
    nlogp = -F.log_softmax(logits, dim=1)

    loss = (positive * nlogp[:, None, 1] + (1 - positive) * nlogp[:, None, 0])

    return loss.mean()

def sigmoid_l1_loss(logits, targets, offset = 0.0, mask=None):
    logp = torch.sigmoid(logits) + offset
    loss = torch.abs(logp-targets)

    if mask is not None:
        w = mask.mean(3, True).mean(2,True)
        w[w==0] = 1
        loss = loss*(mask/w)

    return loss.mean()

def non_maximum_suppression(a):
    ap = F.max_pool2d(a, 3, stride=1, padding=1)
    mask = (a == ap).float().clamp(min=0.0)
    return a * mask

def get_junctions(jloc, joff, topk = 300, th=0):
    height, width = jloc.size(0), jloc.size(1)
    jloc = jloc.reshape(-1)
    joff = joff.reshape(2, -1)

    scores, index = torch.topk(jloc, k=topk)
    y = (index / 128).float() + torch.gather(joff[1], 0, index) + 0.5
    x = (index % 128).float() + torch.gather(joff[0], 0, index) + 0.5

    junctions = torch.stack((x, y)).t()

    return junctions[scores>th], scores[scores>th]

class WireframeParser(nn.Module):
    def __init__(self, cfg):
        super(WireframeParser,self).__init__()
        self.hafm_encoder = HAFMencoder(cfg)

        self.n_dyn_junc = cfg.MODEL.PARSING_HEAD.N_DYN_JUNC
        self.n_dyn_posl = cfg.MODEL.PARSING_HEAD.N_DYN_POSL
        self.n_dyn_negl = cfg.MODEL.PARSING_HEAD.N_DYN_NEGL
        self.n_dyn_othr = cfg.MODEL.PARSING_HEAD.N_DYN_OTHR
        self.n_dyn_othr2= cfg.MODEL.PARSING_HEAD.N_DYN_OTHR2
        self.n_pts0     = cfg.MODEL.PARSING_HEAD.N_PTS0
        self.n_pts1     = cfg.MODEL.PARSING_HEAD.N_PTS1
        self.dim_loi    = cfg.MODEL.PARSING_HEAD.DIM_LOI
        self.dim_fc     = cfg.MODEL.PARSING_HEAD.DIM_FC
        self.n_out_junc = cfg.MODEL.PARSING_HEAD.N_OUT_JUNC
        self.n_out_line = cfg.MODEL.PARSING_HEAD.N_OUT_LINE
        self.use_residual = cfg.MODEL.PARSING_HEAD.USE_RESIDUAL

        self.register_buffer('tspan', torch.linspace(0, 1, self.n_pts0)[None,None,:].cuda())
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.fc1 = nn.Conv2d(256, self.dim_loi, 1)
        self.pool1d = nn.MaxPool1d(self.n_pts0//self.n_pts1, self.n_pts0//self.n_pts1)
        self.fc2 = nn.Sequential(
            nn.Linear(self.dim_loi * self.n_pts1, self.dim_fc),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_fc, self.dim_fc),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_fc, 1),
        )

        self.backbone     = build_backbone(cfg)
    
    def pooling(self, features_per_image, lines_per_im):
        h,w = features_per_image.size(1), features_per_image.size(2)
        U,V = lines_per_im[:,:2], lines_per_im[:,2:]
        sampled_points = U[:,:,None]*self.tspan + V[:,:,None]*(1-self.tspan) -0.5
        sampled_points = sampled_points.permute((0,2,1)).reshape(-1,2)
        px,py = sampled_points[:,0],sampled_points[:,1]
        px0 = px.floor().clamp(min=0, max=w-1)
        py0 = py.floor().clamp(min=0, max=h-1)
        px1 = (px0 + 1).clamp(min=0, max=w-1)
        py1 = (py0 + 1).clamp(min=0, max=h-1)
        px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()

        xp = ((features_per_image[:, py0l, px0l] * (py1-py) * (px1 - px)+ features_per_image[:, py1l, px0l] * (py - py0) * (px1 - px)+ features_per_image[:, py0l, px1l] * (py1 - py) * (px - px0)+ features_per_image[:, py1l, px1l] * (py - py0) * (px - px0)).reshape(128,-1,32)
        ).permute(1,0,2)


        # if self.pool1d is not None:
        xp = self.pool1d(xp)
        features_per_line = xp.view(-1, self.n_pts1*self.dim_loi)
        logits = self.fc2(features_per_line).flatten()
        return logits
    def _forward_test(self, images, annotations = None):
        outputs, features = self.backbone(images)
        extra_info = {}
        loi_features = self.fc1(features)
        output = outputs[0]

        hafm_ang_pred = output[:,:3].sigmoid()
        hafm_dis_pred = output[:,3:4].sigmoid()
        res_pred      = output[:,4:5].sigmoid()
        jloc_pred     = output[:,5:7].softmax(1)[:,1:]
        joff_pred     = output[:,7:9].sigmoid() - 0.5
        batch_size = output.size(0)
        assert batch_size == 1

        if self.use_residual:
            lines_pred = torch.cat(
                [
                self.proposal_lines(hafm_ang_pred[0],
                                    hafm_dis_pred[0],
                                    scale=self.hafm_encoder.dis_th).view(-1,4),
                self.proposal_lines(hafm_ang_pred[0],
                                    hafm_dis_pred[0]+res_pred[0],
                                    scale=self.hafm_encoder.dis_th).view(-1,4),
                self.proposal_lines(hafm_ang_pred[0],
                                    hafm_dis_pred[0]-res_pred[0],
                                    scale=self.hafm_encoder.dis_th).view(-1,4)],dim=0)
        else:
            lines_pred = self.proposal_lines(hafm_ang_pred[0],
                                    hafm_dis_pred[0],
                                    scale=self.hafm_encoder.dis_th).view(-1,4)
        # import pdb; pdb.set_trace()
        jloc_pred_nms = non_maximum_suppression(jloc_pred[0])
        topK = min(300, int((jloc_pred_nms>0.008).float().sum().item()))
        juncs_pred, _ = get_junctions(non_maximum_suppression(jloc_pred[0]),joff_pred[0], topk=topK)
     
        dis_junc_to_end1, idx_junc_to_end1 = torch.sum((lines_pred[:,:2]-juncs_pred[:,None])**2,dim=-1).min(0)
        dis_junc_to_end2, idx_junc_to_end2 = torch.sum((lines_pred[:,2:] - juncs_pred[:, None]) ** 2, dim=-1).min(0)

        idx_junc_to_end_min = torch.min(idx_junc_to_end1,idx_junc_to_end2)
        idx_junc_to_end_max = torch.max(idx_junc_to_end1,idx_junc_to_end2)

        iskeep = (idx_junc_to_end_min < idx_junc_to_end_max)# * (dis_junc_to_end1< 10*10)*(dis_junc_to_end2<10*10)  # *(dis_junc_to_end2<100)

        idx_lines_for_junctions = torch.unique(torch.cat((idx_junc_to_end_min[iskeep,None],idx_junc_to_end_max[iskeep,None]),dim=1),dim=0)

        lines_adjusted = torch.cat((juncs_pred[idx_lines_for_junctions[:,0]], juncs_pred[idx_lines_for_junctions[:,1]]),dim=1)
        scores = self.pooling(loi_features[0],lines_adjusted).sigmoid()

        lines_final = lines_adjusted[scores>0.05]
        score_final = scores[scores>0.05]

        juncs_final = juncs_pred[idx_lines_for_junctions.unique()]
        juncs_score = _[idx_lines_for_junctions.unique()]

        sx = annotations[0]['width']/output.size(3)
        sy = annotations[0]['height']/output.size(3)
        lines_final[:,0] *= sx
        lines_final[:,1] *= sy
        lines_final[:,2] *= sx
        lines_final[:,3] *= sy

        juncs_final[:,0] *= sx
        juncs_final[:,1] *= sy
        
        output = {
            'lines_pred': lines_final,
            'lines_score': score_final,
            'juncs_pred': juncs_final,
            'juncs_score': juncs_score,
            'num_proposals': lines_adjusted.size(0),
            'filename': annotations[0]['filename'],
            'width': annotations[0]['width'],
            'height': annotations[0]['height'],
        }

        return output, extra_info

    def _forward_train(self, images, annotations):
        device = images.device
        targets, metas = self.hafm_encoder(annotations)
        
        outputs, features = self.backbone(images)
        extra_info = {}
        loss_dict = {
            'loss_hafm_ang': 0.0,
            'loss_hafm_dis': 0.0,
            'loss_res': 0.0,
            'loss_jloc': 0.0,
            'loss_joff': 0.0,
            'loss_pos': 0.0,
            'loss_neg': 0.0,
        }
        mask = targets['mask']
        for nstack, output in enumerate(outputs):
            loss_map = torch.mean(F.l1_loss(output[:,:3].sigmoid(), targets['hafm_ang'],reduction='none'),dim=1,keepdim=True)
            loss_dict['loss_hafm_ang']  += torch.mean(loss_map*mask) / torch.mean(mask)
            loss_map = F.l1_loss(output[:,3:4].sigmoid(), targets['hafm_dis'], reduction='none')
            loss_dict['loss_hafm_dis'] += torch.mean(loss_map*mask) /torch.mean(mask)
            loss_residual_map = F.l1_loss(output[:,4:5].sigmoid(), loss_map, reduction='none')
            loss_dict['loss_res'] += torch.mean(loss_residual_map*mask)/torch.mean(mask)
            loss_dict['loss_jloc'] += cross_entropy_loss_for_junction(output[:,5:7], targets['jmap'])
            loss_dict['loss_joff'] += sigmoid_l1_loss(output[:,7:9], targets['joff'], -0.5, targets['jmap'][:,None])
        

        loi_features = self.fc1(features)
        output = outputs[0]

        hafm_ang_pred = output[:,:3].sigmoid()
        hafm_dis_pred = output[:,3:4].sigmoid()
        res_pred      = output[:,4:5].sigmoid()
        jloc_pred     = output[:,5:7].softmax(1)[:,1:]
        joff_pred     = output[:,7:9].sigmoid() - 0.5

        # self.use_residual = False
        # hafm_ang_pred = targets['hafm_ang']
        # hafm_dis_pred = targets['hafm_dis']
        # jloc_pred     = targets['jmap'][:,None]
        # joff_pred     = targets['joff']

        batch_size = output.size(0)

        for i, (hafm_ang_pred_per_im, hafm_dis_pred_per_im, res_pred_per_im, meta) \
            in enumerate(zip(hafm_ang_pred,hafm_dis_pred,res_pred,metas)):
            lines_pred = []
            if self.use_residual:
                for scale in [-1.0,0.0,1.0]:
                    _ = self.proposal_lines(hafm_ang_pred_per_im,
                                            hafm_dis_pred_per_im+scale*res_pred_per_im,
                                            scale=self.hafm_encoder.dis_th).view(-1,4)
                    lines_pred.append(_)
            else:
                lines_pred.append(
                    self.proposal_lines(hafm_ang_pred_per_im,hafm_dis_pred_per_im,scale=self.hafm_encoder.dis_th).view(-1,4))
            lines_pred = torch.cat(lines_pred)
            
            junction_gt = meta['junctions']         
            N = junction_gt.size(0)             
            juncs_pred, _ = get_junctions(
                non_maximum_suppression(jloc_pred[i]),
                joff_pred[i], topk=min(N*2+2,self.n_dyn_junc))

            dis_junc_to_end1, idx_junc_to_end1 = torch.sum((lines_pred[:,:2]-juncs_pred[:,None])**2,dim=-1).min(0)
            dis_junc_to_end2, idx_junc_to_end2 = torch.sum((lines_pred[:, 2:] - juncs_pred[:, None]) ** 2, dim=-1).min(0)

            idx_junc_to_end_min = torch.min(idx_junc_to_end1,idx_junc_to_end2)
            idx_junc_to_end_max = torch.max(idx_junc_to_end1,idx_junc_to_end2)
            # iskeep = (idx_junc_to_end_min < idx_junc_to_end_max)*(dis_junc_to_end1<=10*10)*(dis_junc_to_end2<=10*10)
            iskeep = (idx_junc_to_end_min < idx_junc_to_end_max)
            idx_lines_for_junctions = torch.cat((idx_junc_to_end_min[iskeep,None],idx_junc_to_end_max[iskeep,None]),dim=1).unique(dim=0)

            idx_lines_for_junctions_mirror = torch.cat((idx_lines_for_junctions[:,1,None],idx_lines_for_junctions[:,0,None]),dim=1)
            idx_lines_for_junctions = torch.cat((idx_lines_for_junctions, idx_lines_for_junctions_mirror))
            lines_adjusted = torch.cat((juncs_pred[idx_lines_for_junctions[:,0]], juncs_pred[idx_lines_for_junctions[:,1]]),dim=1)
            cost_, match_ = torch.sum((juncs_pred-junction_gt[:,None])**2,dim=-1).min(0)
            match_[cost_>1.5*1.5] = N

            pos_mat = meta['pos_mat']
            neg_mat = meta['neg_mat']
            labels = pos_mat[match_[idx_lines_for_junctions[:,0]],match_[idx_lines_for_junctions[:,1]]]
            iskeep = torch.zeros_like(labels, dtype= torch.bool)
            cdx = labels.nonzero().flatten()

            if len(cdx) > self.n_dyn_posl:
                perm = torch.randperm(len(cdx),device=device)[:self.n_dyn_posl]
                cdx = cdx[perm]
            
            iskeep[cdx] = 1

            if self.n_dyn_negl > 0:
                cdx = neg_mat[match_[idx_lines_for_junctions[:,0]],match_[idx_lines_for_junctions[:,1]]].nonzero().flatten()
                if len(cdx) > self.n_dyn_negl:
                    perm = torch.randperm(len(cdx), device=device)[:self.n_dyn_negl]
                    cdx = cdx[perm]

                iskeep[cdx] = 1
            
            if self.n_dyn_othr > 0:
                cdx = torch.randint(len(iskeep), (self.n_dyn_othr,), device=device)
                iskeep[cdx] = 1

            if self.n_dyn_othr2 >0 :
                cdx = (labels==0).nonzero().flatten()
                if len(cdx) > self.n_dyn_othr2:
                    perm = torch.randperm(len(cdx), device=device)[:self.n_dyn_othr2]
                    cdx = cdx[perm]
                iskeep[cdx] = 1


            lines_selected = lines_adjusted[iskeep]
            labels_selected = labels[iskeep]

            lines_for_train = torch.cat((lines_selected,meta['lpre']))
            labels_for_train = torch.cat((labels_selected.float(),meta['lpre_label']))

            logits = self.pooling(loi_features[i],lines_for_train)

            loss_ = self.loss(logits, labels_for_train)

            loss_positive = loss_[labels_for_train==1].mean()
            loss_negative = loss_[labels_for_train==0].mean()

            loss_dict['loss_pos'] += loss_positive/batch_size
            loss_dict['loss_neg'] += loss_negative/batch_size

        return loss_dict, extra_info

    def forward(self, images, annotations = None):
        if self.training: 
            assert annotations is not None
            return self._forward_train(images, annotations)
        else:
            with torch.no_grad():
                return self._forward_test(images, annotations)
        

    def proposal_lines(self, hafm_ang, hafm_dis, scale):
        device = hafm_ang.device
        height, width = hafm_ang.size(1), hafm_ang.size(2)
        _y = torch.arange(0,height,device=device).float()
        _x = torch.arange(0,width,device=device).float()

        y0,x0 = torch.meshgrid(_y,_x)
        md_ = (hafm_ang[0]-0.5)*np.pi*2
        st_ = hafm_ang[1]*np.pi/2
        ed_ = -hafm_ang[2]*np.pi/2

        cs_md = torch.cos(md_)
        ss_md = torch.sin(md_)

        cs_st = torch.cos(st_).clamp(min=1e-3)
        ss_st = torch.sin(st_).clamp(min=1e-3)

        cs_ed = torch.cos(ed_).clamp(min=1e-3)
        ss_ed = torch.sin(ed_).clamp(max=-1e-3)

        # x_standard = torch.ones_like(cs_st)

        y_st = ss_st/cs_st
        y_ed = ss_ed/cs_ed
        # import pdb
        # pdb.set_trace()
        x_st_rotated =  (cs_md - ss_md*y_st)*hafm_dis[0]*scale
        y_st_rotated =  (ss_md + cs_md*y_st)*hafm_dis[0]*scale

        x_ed_rotated =  (cs_md - ss_md*y_ed)*hafm_dis[0]*scale
        y_ed_rotated = (ss_md + cs_md*y_ed)*hafm_dis[0]*scale

        x_st_final = (x_st_rotated + x0).clamp(min=0,max=width-1)
        y_st_final = (y_st_rotated + y0).clamp(min=0,max=height-1)

        x_ed_final = (x_ed_rotated + x0).clamp(min=0,max=width-1)
        y_ed_final = (y_ed_rotated + y0).clamp(min=0,max=height-1)

        lines = torch.stack((x_st_final,y_st_final,x_ed_final,y_ed_final)).permute((1,2,0))
        return lines
