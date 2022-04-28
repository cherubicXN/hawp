import torch
from torch import nn
from parsing.backbones import build_backbone
from parsing.encoder.hafm import HAFMencoder
# from epnet.structures.linelist_ops import linesegment_distance
import torch.nn.functional as F
import matplotlib.pyplot as plt
import  numpy as np
import time

PRETRAINED = {
    'url': 'https://github.com/cherubicXN/hawp-torchhub/releases/download/0.1/model-hawp-hg-5d31f70.pth',
    'md5': '5d31f70a6c2477ea7b24e7da96e7b97d',
}

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
    height, width = jloc.size(1), jloc.size(2)
    jloc = jloc.reshape(-1)
    joff = joff.reshape(2, -1)

    scores, index = torch.topk(jloc, k=topk)
    y = (index // width).float() + torch.gather(joff[1], 0, index) + 0.5
    x = (index % width).float() + torch.gather(joff[0], 0, index) + 0.5

    junctions = torch.stack((x, y)).t()

    return junctions[scores>th], scores[scores>th]

class WireframeDetector(nn.Module):
    def __init__(self, cfg):
        super(WireframeDetector, self).__init__()
        self.hafm_encoder = HAFMencoder(cfg)
        self.backbone = build_backbone(cfg)

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
        # self.
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
        self.train_step = 0

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

        xp = ((features_per_image[:, py0l, px0l] * (py1-py) * (px1 - px)+ features_per_image[:, py1l, px0l] * (py - py0) * (px1 - px)+ features_per_image[:, py0l, px1l] * (py1 - py) * (px - px0)+ features_per_image[:, py1l, px1l] * (py - py0) * (px - px0)).reshape(self.dim_loi,-1,self.n_pts0)
        ).permute(1,0,2)


        # if self.pool1d is not None:
        xp = self.pool1d(xp)
        features_per_line = xp.view(-1, self.n_pts1*self.dim_loi)
        logits = self.fc2(features_per_line).flatten()
        return logits

    def forward(self, images, annotations = None):
        if self.training:
            return self.forward_train(images, annotations=annotations)
        else:
            return self.forward_test(images, annotations=annotations)

    def forward_test(self, images, annotations = None):
        device = images.device

        extra_info = {
            'time_backbone': 0.0,
            'time_proposal': 0.0,
            'time_matching': 0.0,
            'time_verification': 0.0,
        }

        extra_info['time_backbone'] = time.time()
        outputs, features = self.backbone(images)

        loi_features = self.fc1(features)
        output = outputs[0]
        md_pred = output[:,:3].sigmoid()
        # dis_pred = targets['dis']
        dis_pred = output[:,3:4].sigmoid()
        res_pred = output[:,4:5].sigmoid()
        jloc_pred= output[:,5:7].softmax(1)[:,1:]
        joff_pred= output[:,7:9].sigmoid() - 0.5
        extra_info['time_backbone'] = time.time() - extra_info['time_backbone']


        batch_size = md_pred.size(0)
        assert batch_size == 1

        extra_info['time_proposal'] = time.time()
        if self.use_residual:
            lines_pred = self.proposal_lines_new(md_pred[0],dis_pred[0],res_pred[0]).view(-1,4)
        else:
            lines_pred = self.proposal_lines_new(md_pred[0], dis_pred[0], None).view(-1, 4)

        jloc_pred_nms = non_maximum_suppression(jloc_pred[0])
        topK = min(300, int((jloc_pred_nms>0.008).float().sum().item()))

        juncs_pred, _ = get_junctions(non_maximum_suppression(jloc_pred[0]),joff_pred[0], topk=topK)
        extra_info['time_proposal'] = time.time() - extra_info['time_proposal']
        extra_info['time_matching'] = time.time()
        dis_junc_to_end1, idx_junc_to_end1 = torch.sum((lines_pred[:,:2]-juncs_pred[:,None])**2,dim=-1).min(0)
        dis_junc_to_end2, idx_junc_to_end2 = torch.sum((lines_pred[:,2:] - juncs_pred[:, None]) ** 2, dim=-1).min(0)

        idx_junc_to_end_min = torch.min(idx_junc_to_end1,idx_junc_to_end2)
        idx_junc_to_end_max = torch.max(idx_junc_to_end1,idx_junc_to_end2)

        iskeep = (idx_junc_to_end_min < idx_junc_to_end_max)# * (dis_junc_to_end1< 10*10)*(dis_junc_to_end2<10*10)  # *(dis_junc_to_end2<100)

        idx_lines_for_junctions = torch.unique(
            torch.cat((idx_junc_to_end_min[iskeep,None],idx_junc_to_end_max[iskeep,None]),dim=1),
            dim=0)
        lines_adjusted = torch.cat((juncs_pred[idx_lines_for_junctions[:,0]], juncs_pred[idx_lines_for_junctions[:,1]]),dim=1)
        extra_info['time_matching'] = time.time() - extra_info['time_matching']

        extra_info['time_verification'] = time.time()
        scores = self.pooling(loi_features[0],lines_adjusted).sigmoid()

        lines_final = lines_adjusted[scores>0.05]
        score_final = scores[scores>0.05]

        sarg = torch.argsort(score_final,descending=True)

        juncs_final = juncs_pred[idx_lines_for_junctions.unique()]
        juncs_score = _[idx_lines_for_junctions.unique()]

        extra_info['time_verification'] = time.time() - extra_info['time_verification']

        sx = annotations[0]['width']/output.size(3)
        sy = annotations[0]['height']/output.size(2)

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
    def forward_train(self, images, annotations = None):
        device = images.device

        targets , metas = self.hafm_encoder(annotations)

        self.train_step += 1

        outputs, features = self.backbone(images)

        loss_dict = {
            'loss_md': 0.0,
            'loss_dis': 0.0,
            'loss_res': 0.0,
            'loss_jloc': 0.0,
            'loss_joff': 0.0,
            'loss_pos': 0.0,
            'loss_neg': 0.0,
        }


        mask = targets['mask']
        if targets is not None:
            for nstack, output in enumerate(outputs):
                loss_map = torch.mean(F.l1_loss(output[:,:3].sigmoid(), targets['md'],reduction='none'),dim=1,keepdim=True)
                loss_dict['loss_md']  += torch.mean(loss_map*mask) / torch.mean(mask)
                loss_map = F.l1_loss(output[:,3:4].sigmoid(), targets['dis'], reduction='none')
                loss_dict['loss_dis'] += torch.mean(loss_map*mask) /torch.mean(mask)
                loss_residual_map = F.l1_loss(output[:,4:5].sigmoid(), loss_map, reduction='none')
                loss_dict['loss_res'] += torch.mean(loss_residual_map*mask)/torch.mean(mask)
                loss_dict['loss_jloc'] += cross_entropy_loss_for_junction(output[:,5:7], targets['jloc'])
                loss_dict['loss_joff'] += sigmoid_l1_loss(output[:,7:9], targets['joff'], -0.5, targets['jloc'])

        loi_features = self.fc1(features)
        output = outputs[0]
        md_pred = output[:,:3].sigmoid()
        dis_pred = output[:,3:4].sigmoid()
        res_pred = output[:,4:5].sigmoid()
        jloc_pred= output[:,5:7].softmax(1)[:,1:]
        joff_pred= output[:,7:9].sigmoid() - 0.5

        lines_batch = []
        extra_info = {
        }

        batch_size = md_pred.size(0)

        for i, (md_pred_per_im, dis_pred_per_im,res_pred_per_im,meta) in enumerate(zip(md_pred, dis_pred,res_pred,metas)):
            lines_pred = []
            if self.use_residual:
                for scale in [-1.0,0.0,1.0]:
                    _ = self.proposal_lines(md_pred_per_im, dis_pred_per_im+scale*res_pred_per_im).view(-1, 4)
                    lines_pred.append(_)
            else:
                lines_pred.append(self.proposal_lines(md_pred_per_im, dis_pred_per_im).view(-1, 4))
            lines_pred = torch.cat(lines_pred)
            junction_gt = meta['junc']
            N = junction_gt.size(0)

            juncs_pred, _ = get_junctions(non_maximum_suppression(jloc_pred[i]),joff_pred[i], topk=min(N*2+2,self.n_dyn_junc))
            dis_junc_to_end1, idx_junc_to_end1 = torch.sum((lines_pred[:,:2]-juncs_pred[:,None])**2,dim=-1).min(0)
            dis_junc_to_end2, idx_junc_to_end2 = torch.sum((lines_pred[:, 2:] - juncs_pred[:, None]) ** 2, dim=-1).min(0)

            idx_junc_to_end_min = torch.min(idx_junc_to_end1,idx_junc_to_end2)
            idx_junc_to_end_max = torch.max(idx_junc_to_end1,idx_junc_to_end2)
            iskeep = idx_junc_to_end_min<idx_junc_to_end_max
            idx_lines_for_junctions = torch.cat((idx_junc_to_end_min[iskeep,None],idx_junc_to_end_max[iskeep,None]),dim=1).unique(dim=0)
            idx_lines_for_junctions_mirror = torch.cat((idx_lines_for_junctions[:,1,None],idx_lines_for_junctions[:,0,None]),dim=1)
            idx_lines_for_junctions = torch.cat((idx_lines_for_junctions, idx_lines_for_junctions_mirror))
            lines_adjusted = torch.cat((juncs_pred[idx_lines_for_junctions[:,0]], juncs_pred[idx_lines_for_junctions[:,1]]),dim=1)

            cost_, match_ = torch.sum((juncs_pred-junction_gt[:,None])**2,dim=-1).min(0)
            match_[cost_>1.5*1.5] = N
            Lpos = meta['Lpos']
            Lneg = meta['Lneg']
            labels = Lpos[match_[idx_lines_for_junctions[:,0]],match_[idx_lines_for_junctions[:,1]]]

            iskeep = torch.zeros_like(labels, dtype= torch.bool)
            cdx = labels.nonzero().flatten()

            if len(cdx) > self.n_dyn_posl:
                perm = torch.randperm(len(cdx),device=device)[:self.n_dyn_posl]
                cdx = cdx[perm]

            iskeep[cdx] = 1

            if self.n_dyn_negl > 0:
                cdx = Lneg[match_[idx_lines_for_junctions[:,0]],match_[idx_lines_for_junctions[:,1]]].nonzero().flatten()

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

    def proposal_lines(self, md_maps, dis_maps, scale=5.0):
        """

        :param md_maps: 3xhxw, the range should be (0,1) for every element
        :param dis_maps: 1xhxw
        :return:
        """
        device = md_maps.device
        height, width = md_maps.size(1), md_maps.size(2)
        _y = torch.arange(0,height,device=device).float()
        _x = torch.arange(0,width, device=device).float()

        y0,x0 = torch.meshgrid(_y,_x)
        md_ = (md_maps[0]-0.5)*np.pi*2
        st_ = md_maps[1]*np.pi/2
        ed_ = -md_maps[2]*np.pi/2

        cs_md = torch.cos(md_)
        ss_md = torch.sin(md_)

        cs_st = torch.cos(st_).clamp(min=1e-3)
        ss_st = torch.sin(st_).clamp(min=1e-3)

        cs_ed = torch.cos(ed_).clamp(min=1e-3)
        ss_ed = torch.sin(ed_).clamp(max=-1e-3)

        x_standard = torch.ones_like(cs_st)

        y_st = ss_st/cs_st
        y_ed = ss_ed/cs_ed

        x_st_rotated =  (cs_md - ss_md*y_st)*dis_maps[0]*scale
        y_st_rotated =  (ss_md + cs_md*y_st)*dis_maps[0]*scale

        x_ed_rotated =  (cs_md - ss_md*y_ed)*dis_maps[0]*scale
        y_ed_rotated = (ss_md + cs_md*y_ed)*dis_maps[0]*scale

        x_st_final = (x_st_rotated + x0).clamp(min=0,max=width-1)
        y_st_final = (y_st_rotated + y0).clamp(min=0,max=height-1)

        x_ed_final = (x_ed_rotated + x0).clamp(min=0,max=width-1)
        y_ed_final = (y_ed_rotated + y0).clamp(min=0,max=height-1)

        lines = torch.stack((x_st_final,y_st_final,x_ed_final,y_ed_final)).permute((1,2,0))

        return  lines#, normals

    def proposal_lines_new(self, md_maps, dis_maps, residual_maps, scale=5.0):
        """

        :param md_maps: 3xhxw, the range should be (0,1) for every element
        :param dis_maps: 1xhxw
        :return:
        """
        device = md_maps.device
        sign_pad     = torch.tensor([-1,0,1],device=device,dtype=torch.float32).reshape(3,1,1)

        if residual_maps is None:
            dis_maps_new = dis_maps.repeat((1,1,1))
        else:
            dis_maps_new = dis_maps.repeat((3,1,1))+sign_pad*residual_maps.repeat((3,1,1))
        height, width = md_maps.size(1), md_maps.size(2)
        _y = torch.arange(0,height,device=device).float()
        _x = torch.arange(0,width, device=device).float()

        y0,x0 = torch.meshgrid(_y,_x)
        md_ = (md_maps[0]-0.5)*np.pi*2
        st_ = md_maps[1]*np.pi/2
        ed_ = -md_maps[2]*np.pi/2

        cs_md = torch.cos(md_)
        ss_md = torch.sin(md_)

        cs_st = torch.cos(st_).clamp(min=1e-3)
        ss_st = torch.sin(st_).clamp(min=1e-3)

        cs_ed = torch.cos(ed_).clamp(min=1e-3)
        ss_ed = torch.sin(ed_).clamp(max=-1e-3)

        y_st = ss_st/cs_st
        y_ed = ss_ed/cs_ed

        x_st_rotated = (cs_md-ss_md*y_st)[None]*dis_maps_new*scale
        y_st_rotated =  (ss_md + cs_md*y_st)[None]*dis_maps_new*scale

        x_ed_rotated =  (cs_md - ss_md*y_ed)[None]*dis_maps_new*scale
        y_ed_rotated = (ss_md + cs_md*y_ed)[None]*dis_maps_new*scale

        x_st_final = (x_st_rotated + x0[None]).clamp(min=0,max=width-1)
        y_st_final = (y_st_rotated + y0[None]).clamp(min=0,max=height-1)

        x_ed_final = (x_ed_rotated + x0[None]).clamp(min=0,max=width-1)
        y_ed_final = (y_ed_rotated + y0[None]).clamp(min=0,max=height-1)

        lines = torch.stack((x_st_final,y_st_final,x_ed_final,y_ed_final)).permute((1,2,3,0))

        # normals = torch.stack((cs_md,ss_md)).permute((1,2,0))

        return  lines#, normals

def get_hawp_model(pretrained = False):
    from parsing.config import cfg
    import os
    model = WireframeDetector(cfg)
    if pretrained:
        url = PRETRAINED.get('url')
        hubdir = torch.hub.get_dir()
        filename = os.path.basename(url)
        dst = os.path.join(hubdir,filename)
        state_dict = torch.hub.load_state_dict_from_url(url,dst)
        model.load_state_dict(state_dict)
        model = model.eval()
        return model
    return model

        
