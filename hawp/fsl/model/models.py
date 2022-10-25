import torch
from torch import nn
from hawp.fsl.backbones import build_backbone
from collections import defaultdict
import torch.nn.functional as F
import matplotlib.pyplot as plt
import  numpy as np
import time
from scipy.optimize import linear_sum_assignment

from .hafm import HAFMencoder
from .losses import cross_entropy_loss_for_junction, sigmoid_l1_loss, sigmoid_focal_loss
from .misc import non_maximum_suppression, get_junctions, plot_lines

def argsort2d(arr):
    return np.dstack(np.unravel_index(np.argsort(arr.ravel()), arr.shape))[0]

def nms_j(heatmap, delta=1):
    DX = [0, 0, 1, -1, 1, 1, -1, -1]
    DY = [1, -1, 0, 0, 1, -1, 1, -1]
    heatmap = heatmap.copy()
    disable = np.zeros_like(heatmap, dtype=np.bool)
    for x, y in argsort2d(heatmap):
        for dx, dy in zip(DX, DY):
            xp, yp = x + dx, y + dy
            if not (0 <= xp < heatmap.shape[0] and 0 <= yp < heatmap.shape[1]):
                continue
            if heatmap[x, y] >= heatmap[xp, yp]:
                disable[xp, yp] = True
    heatmap[disable] *= 0.6
    return heatmap
def post_jheatmap(heatmap, offset=None, delta=1):
    heatmap = nms_j(heatmap, delta=delta)
    # only select the best 1000 junctions for efficiency
    v0 = argsort2d(-heatmap)[:1000]
    confidence = -np.sort(-heatmap.ravel())[:1000]
    # v0 = argsort2d(-heatmap)[:250]
    # confidence = -np.sort(-heatmap.ravel())[:250]
    keep_id = np.where(confidence >= 1e-2)[0]
    if len(keep_id) == 0:
        return np.zeros((0, 3))

    confidence = confidence[keep_id]
    if offset is not None:
        v0 = np.array([v + offset[:, v[0], v[1]] for v in v0])
    v0 = v0[keep_id] + 0.5
    v0 = np.hstack((v0, confidence[:, np.newaxis]))
    return v0

def add_argument_with_cfg(parser, cfg, arg_name, cfg_name, help, mapping):
    
    parser.add_argument('--{}'.format(arg_name.replace('_','-')), 
        default = eval('cfg.{}'.format(cfg_name)),
        type = type(eval('cfg.{}'.format(cfg_name))),
        help = help
    )
    mapping[arg_name] = cfg_name

class WireframeDetector(nn.Module):
    def cli(self, cfg, argparser):
        cfg_mapping = {}
        sampling_parser = argparser.add_argument_group(title = 'sampling specification')
        add_argument_lambda = lambda arg_name, cfg_name, help: add_argument_with_cfg(sampling_parser, cfg, arg_name, cfg_name, help, mapping=cfg_mapping)

        add_argument_lambda('num_dyn_junctions','MODEL.PARSING_HEAD.N_DYN_JUNC', help = '[train] number of dynamic junctions')
        add_argument_lambda('num_dyn_positive_lines', 'MODEL.PARSING_HEAD.N_DYN_POSL', help ='[train] number of dynamic positive lines')
        add_argument_lambda('num_dyn_negative_lines','MODEL.PARSING_HEAD.N_DYN_NEGL', help='[train] number of dynamic negative lines')
        add_argument_lambda('num_dyn_natural_lines', 'MODEL.PARSING_HEAD.N_DYN_OTHR2', help='[train] number of dynamic line samples from the natural selection')

        matching_parser = argparser.add_argument_group(title = 'matching specification')
        add_argument_lambda = lambda arg_name, cfg_name, help: add_argument_with_cfg(matching_parser, cfg, arg_name, cfg_name, help, mapping=cfg_mapping)

        add_argument_lambda('j2l_threshold','MODEL.PARSING_HEAD.J2L_THRESHOLD', help='[all] the matching distance (in pixels^2) between the junctions and the learned lines')
        add_argument_lambda('jmatch_threshold', 'MODEL.PARSING_HEAD.JMATCH_THRESHOLD', help='[train] the matching distance (in pixels) between the predicted and grountruth junctions')

        loi_parser = argparser.add_argument_group(title = 'LOI-pooling specification')
        add_argument_lambda = lambda arg_name, cfg_name, help: add_argument_with_cfg(loi_parser, cfg, arg_name, cfg_name, help, mapping=cfg_mapping)
        add_argument_lambda('num_points', 'MODEL.LOI_POOLING.NUM_POINTS', help='[train] the number of sampling points')
        add_argument_lambda('dim_junction', 'MODEL.LOI_POOLING.DIM_JUNCTION_FEATURE', help='[train] the dim of junction features')
        add_argument_lambda('dim_edge', 'MODEL.LOI_POOLING.DIM_EDGE_FEATURE', help='[train] the dim of edge features')
        add_argument_lambda('dim_fc', 'MODEL.LOI_POOLING.DIM_FC', help='[train] the dim of fc features')

        hafm_parser = argparser.add_argument_group(title = 'Line proposal specification')
        add_argument_lambda = lambda arg_name, cfg_name, help: add_argument_with_cfg(hafm_parser, cfg, arg_name, cfg_name, help, mapping=cfg_mapping)
        add_argument_lambda('num_residuals', 'MODEL.PARSING_HEAD.USE_RESIDUAL', help='[all] the number of distance residuals')
        self.cfg_mapping = cfg_mapping
        
    def configure(self, cfg, args):
        configure_list = []
        for key, value in self.cfg_mapping.items():
            if getattr(args,key) != eval('cfg.'+value):
                configure_list.extend([value,getattr(args,key)])
        cfg.merge_from_list(configure_list)
    def __init__(self, cfg):
        super(WireframeDetector, self).__init__()
        self.hafm_encoder = HAFMencoder(cfg)
        self.backbone = build_backbone(cfg)

        self.n_dyn_junc = cfg.MODEL.PARSING_HEAD.N_DYN_JUNC
        self.n_dyn_posl = cfg.MODEL.PARSING_HEAD.N_DYN_POSL
        self.n_dyn_negl = cfg.MODEL.PARSING_HEAD.N_DYN_NEGL
        self.n_dyn_othr = cfg.MODEL.PARSING_HEAD.N_DYN_OTHR
        self.n_dyn_othr2= cfg.MODEL.PARSING_HEAD.N_DYN_OTHR2
        self.topk_junctions = 300
        #Matcher
        self.j2l_threshold = cfg.MODEL.PARSING_HEAD.J2L_THRESHOLD
        self.jmatch_threshold = cfg.MODEL.PARSING_HEAD.JMATCH_THRESHOLD

        # LOI POOLING
        self.n_pts0     = cfg.MODEL.LOI_POOLING.NUM_POINTS
        self.dim_junction_feature    = cfg.MODEL.LOI_POOLING.DIM_JUNCTION_FEATURE
        self.dim_edge_feature = cfg.MODEL.LOI_POOLING.DIM_EDGE_FEATURE
        self.dim_fc     = cfg.MODEL.LOI_POOLING.DIM_FC


        self.n_out_junc = cfg.MODEL.PARSING_HEAD.N_OUT_JUNC
        self.n_out_line = cfg.MODEL.PARSING_HEAD.N_OUT_LINE
        self.use_residual = int(cfg.MODEL.PARSING_HEAD.USE_RESIDUAL)

        self.register_buffer('tspan', torch.linspace(0, 1, self.n_pts0)[None,None,:].cuda())
        
        assert cfg.MODEL.LOI_POOLING.TYPE in ['softmax', 'sigmoid']
        assert cfg.MODEL.LOI_POOLING.ACTIVATION in ['relu', 'gelu']

        self.loi_cls_type = cfg.MODEL.LOI_POOLING.TYPE
        self.loi_layer_norm = cfg.MODEL.LOI_POOLING.LAYER_NORM
        self.loi_activation = nn.ReLU if cfg.MODEL.LOI_POOLING.ACTIVATION == 'relu' else nn.GELU        

        self.fc1 = nn.Conv2d(256, self.dim_junction_feature, 1)

        self.fc3 = nn.Conv2d(256, self.dim_edge_feature, 1)
        self.fc4 = nn.Conv2d(256, self.dim_edge_feature, 1)

        self.regional_head = nn.Conv2d(256, 1, 1)
        fc2 = [nn.Linear(self.dim_junction_feature*2 + (self.n_pts0-2)*self.dim_edge_feature*2, self.dim_fc),
                # self.loi_activation(),
        ]
        for i in range(2):
            fc2.append(self.loi_activation())
            fc2.append(nn.Linear(self.dim_fc,self.dim_fc))

        
        self.fc2 = nn.Sequential(*fc2)
        self.fc2_res = nn.Sequential(nn.Linear(2*(self.n_pts0-2)*self.dim_edge_feature, self.dim_fc),self.loi_activation())

        self.line_mlp = nn.Sequential(
            nn.Linear((self.n_pts0-2)*self.dim_edge_feature,128),
            nn.ReLU(True),
            nn.Linear(128,32),nn.ReLU(True),
            nn.Linear(32,1)
        )

        if self.loi_cls_type == 'softmax':
            self.fc2_head = nn.Linear(self.dim_fc, 2)
            self.loss = nn.CrossEntropyLoss(reduction='none')
        elif self.loi_cls_type == 'sigmoid':
            self.fc2_head = nn.Linear(self.dim_fc, 1)
            self.loss = nn.BCEWithLogitsLoss(reduction='none')
        else:
            raise NotImplementError()

        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.train_step = 0

    def bilinear_sampling(self, features, points):
        h,w = features.size(1), features.size(2)
        px, py = points[:,0], points[:,1]
        
        px0 = px.floor().clamp(min=0, max=w-1)
        py0 = py.floor().clamp(min=0, max=h-1)
        px1 = (px0 + 1).clamp(min=0, max=w-1)
        py1 = (py0 + 1).clamp(min=0, max=h-1)
        px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()
        xp = features[:, py0l, px0l] * (py1-py) * (px1 - px)+ features[:, py1l, px0l] * (py - py0) * (px1 - px)+ features[:, py0l, px1l] * (py1 - py) * (px - px0)+ features[:, py1l, px1l] * (py - py0) * (px - px0)

        return xp
    
    def get_line_points(self, lines_per_im):
        U,V = lines_per_im[:,:2], lines_per_im[:,2:]
        sampled_points = U[:,:,None]*self.tspan + V[:,:,None]*(1-self.tspan) -0.5
        return sampled_points
    
    def compute_loi_features(self, features_per_image, lines_per_im):
        num_channels = features_per_image.shape[0]
        h,w = features_per_image.size(1), features_per_image.size(2)
        U,V = lines_per_im[:,:2], lines_per_im[:,2:]
        tspan = self.tspan[...,1:-1]
        sampled_points = U[:,:,None]*tspan + V[:,:,None]*(1-tspan) -0.5

        sampled_points = sampled_points.permute((0,2,1)).reshape(-1,2)
        px,py = sampled_points[:,0],sampled_points[:,1]
        px0 = px.floor().clamp(min=0, max=w-1)
        py0 = py.floor().clamp(min=0, max=h-1)
        px1 = (px0 + 1).clamp(min=0, max=w-1)
        py1 = (py0 + 1).clamp(min=0, max=h-1)
        px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()
        xp = features_per_image[:, py0l, px0l] * (py1-py) * (px1 - px)+ features_per_image[:, py1l, px0l] * (py - py0) * (px1 - px)+ features_per_image[:, py0l, px1l] * (py1 - py) * (px - px0)+ features_per_image[:, py1l, px1l] * (py - py0) * (px - px0)
        xp = xp.reshape(features_per_image.shape[0],-1,tspan.numel()).permute(1,0,2).contiguous()

        return xp.flatten(1)
    def pooling(self, features_per_line):
        
        if self.training:
            logits = self.fc2(features_per_line)
            return logits
        
        if self.loi_cls_type == 'softmax':
            return self.fc2(features_per_line).softmax(dim=-1)[:,1]
        else:
            return self.fc2(features_per_line).sigmoid()[:,0]

    def forward(self, images, annotations = None, targets = None):
        if self.training:
            return self.forward_train(images, annotations=annotations)
        else:
            return self.forward_test(images, annotations=annotations)
        
    def wireframe_matcher(self, juncs_pred, lines_pred, is_train=False,return_index=False):
        cost1 = torch.sum((lines_pred[:,:2]-juncs_pred[:,None])**2,dim=-1)
        cost2 = torch.sum((lines_pred[:,2:]-juncs_pred[:,None])**2,dim=-1)
        
        dis1, idx_junc_to_end1 = cost1.min(dim=0)
        dis2, idx_junc_to_end2 = cost2.min(dim=0)
        length = torch.sum((lines_pred[:,:2]-lines_pred[:,2:])**2,dim=-1)


        idx_junc_to_end_min = torch.min(idx_junc_to_end1,idx_junc_to_end2)
        idx_junc_to_end_max = torch.max(idx_junc_to_end1,idx_junc_to_end2)

        iskeep = idx_junc_to_end_min < idx_junc_to_end_max
        if self.j2l_threshold>0:
            iskeep *= (dis1<self.j2l_threshold)*(dis2<self.j2l_threshold)
        
        idx_lines_for_junctions = torch.stack((idx_junc_to_end_min[iskeep],idx_junc_to_end_max[iskeep]),dim=1)#.unique(dim=0)
        
        idx_lines_for_junctions, inverse = torch.unique(idx_lines_for_junctions,sorted=True,return_inverse=True,dim=0)

        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        perm = inverse.new_empty(idx_lines_for_junctions.size(0)).scatter_(0, inverse, perm)
        lines_init = lines_pred[iskeep][perm]
        if is_train:
            idx_lines_for_junctions_mirror = torch.cat((idx_lines_for_junctions[:,1,None],idx_lines_for_junctions[:,0,None]),dim=1)
            idx_lines_for_junctions = torch.cat((idx_lines_for_junctions, idx_lines_for_junctions_mirror))
        lines_adjusted = juncs_pred[idx_lines_for_junctions].reshape(-1,4)
        
        if return_index:
            return lines_adjusted, lines_init, perm, idx_lines_for_junctions
        else:
            return lines_adjusted, lines_init, perm
    
    def junction_detection(self, images, annotations = None):
        device = images.device

        extra_info = {
            'time_backbone': 0.0,
            'time_proposal': 0.0,
            'time_matching': 0.0,
            'time_verification': 0.0,
        }

        extra_info['time_backbone'] = time.time()
        outputs, features = self.backbone(images)
        output = outputs[0]
        jloc_pred= output[:,5:7].softmax(1)[:,1:]
        jloc_logits = output[:,5:7].softmax(1)
        joff_pred= output[:,7:9].sigmoid()-0.5 

        width = annotations[0]['width']
        height = annotations[0]['height']

        junctions = post_jheatmap(jloc_pred[0,0].cpu().numpy(),offset=joff_pred[0,[1,0]].cpu().numpy())
        # junctions = post_jheatmap(jloc_pred[0,0].cpu().numpy())
        scores = junctions[:,-1]
        junctions = junctions[:,:2]

        junctions_xy = junctions[:,[1,0]]
        # plt.imshow(images[0,0].cpu())
        # plt.plot(junctions_xy[:,0]*4,junctions_xy[:,1]*4,'r.')
        # plt.show()
        # import pdb; pdb.set_trace()
        junctions_xy[:,0] *= float(width/features.shape[-1])
        junctions_xy[:,1] *= float(height/features.shape[-2])
        return {
            'filename': annotations[0]['filename'],
            'juncs_pred': junctions_xy,
            'juncs_score': scores,
            'width': width,
            'height': height,
        }

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
        loi_features_thin = self.fc3(features)
        
        loi_features_aux = self.fc4(features)
        output = outputs[0]
        md_pred = output[:,:3].sigmoid()
        dis_pred = output[:,3:4].sigmoid()
        res_pred = output[:,4:5].sigmoid()
        jloc_pred= output[:,5:7].softmax(1)[:,1:]
        jloc_logits = output[:,5:7].softmax(1)
        joff_pred= output[:,7:9].sigmoid() - 0.5

        extra_info['time_backbone'] = time.time() - extra_info['time_backbone']

        batch_size = md_pred.size(0)
        assert batch_size == 1
        
        extra_info['time_proposal'] = time.time()
        
        lines_pred = self.hafm_decoding(md_pred, dis_pred, res_pred if self.use_residual else None, scale=self.hafm_encoder.dis_th)
        
        lines_pred = lines_pred.reshape(-1,4)
    
        jloc_pred_nms = non_maximum_suppression(jloc_pred[0])

        topK = min(self.topk_junctions, int((jloc_pred_nms>0.008).float().sum().item()))
        
        juncs_pred, _ = get_junctions(non_maximum_suppression(jloc_pred[0]),joff_pred[0], topk=topK,th=0.008)

        extra_info['time_proposal'] = time.time() - extra_info['time_proposal']
        extra_info['time_matching'] = time.time()
        
        
        grid = torch.stack(torch.meshgrid(torch.arange(loi_features.shape[2],device=device),torch.arange(loi_features.shape[3],device=device),indexing='xy'),dim=-1).unsqueeze(0).repeat(2*self.use_residual+1,1,1,1).reshape(-1,2)
        
        lines_adjusted, lines_init, perm, unique_indices = self.wireframe_matcher(juncs_pred, lines_pred,return_index=True)
        extra_info['time_matching'] = time.time() - extra_info['time_matching']
        extra_info['time_verification'] = time.time()

        e1_features = self.bilinear_sampling(loi_features[0], lines_adjusted[:,:2]-0.5).t()
        e2_features = self.bilinear_sampling(loi_features[0], lines_adjusted[:,2:]-0.5).t()
        f1 = self.compute_loi_features(loi_features_thin[0],lines_adjusted)
        f2 = self.compute_loi_features(loi_features_aux[0],lines_init)

        line_features = torch.cat((e1_features,e2_features,f1,f2),dim=-1)
        logits = self.fc2_head(self.fc2(line_features)+self.fc2_res(torch.cat((f1,f2),dim=-1)))

        if self.loi_cls_type == 'softmax':
            scores = logits.softmax(dim=-1)[:,1]
        else:
            scores = logits.sigmoid()[:,0]
        
        sarg = torch.argsort(scores,descending=True)

        lines_final = lines_adjusted[sarg]
        score_final = scores[sarg]
        lines_before = lines_init[sarg]
            
        num_detection = min((score_final>0.0).sum(),1000)
        lines_final = lines_final[:num_detection]
        lines_before = lines_before[:num_detection]
        score_final = score_final[:num_detection]

        # unique_indices = unique_indices.unique()
        juncs_final = juncs_pred
        juncs_score = _
        # juncs_score = _#[idx_lines_for_junctions.unique()]

        extra_info['time_verification'] = time.time() - extra_info['time_verification']

        sx = annotations[0]['width']/output.size(3)
        sy = annotations[0]['height']/output.size(2)
        line_scale_vec = torch.tensor([sx,sy,sx,sy],dtype=torch.float32,device=device).reshape(-1,4)

        lines_final *= line_scale_vec

        lines_before *= line_scale_vec

        juncs_final *= line_scale_vec[:,:2]
        
        output = {
            'lines_pred': lines_final,
            'lines_init': lines_before,
            'lines_score': score_final,
            'juncs_pred': juncs_final,
            'juncs_score': juncs_score,
            'num_proposals': lines_adjusted.size(0),
            'filename': annotations[0]['filename'],
            'width': annotations[0]['width'],
            'height': annotations[0]['height'],
        }

        return output, extra_info

    def focal_loss(self,input, target, gamma=2.0):
        prob = F.softmax(input, 1) 
        ce_loss = F.cross_entropy(input, target,  reduction='none')
        p_t = prob[:,1] * target + prob[:,0] * (1 - target)
        loss = ce_loss * ((1 - p_t) ** gamma)
        return loss
    
    def refinement_train(self, lines_batch, jloc_pred, joff_pred, loi_features, loi_features_thin, loi_features_aux, metas):
        loss_dict = defaultdict(float)
        extra_info = defaultdict(float)
        batch_size = lines_batch.shape[0]
        device = lines_batch.device
        resinds = torch.arange(-self.use_residual,self.use_residual+1,device=device).reshape(-1,1,1).repeat(1,lines_batch.shape[2],lines_batch.shape[3]).reshape(-1)

        grid = torch.stack(torch.meshgrid(torch.arange(lines_batch.shape[2],device=device),torch.arange(lines_batch.shape[3],device=device),indexing='xy'),dim=-1).unsqueeze(0).repeat(2*self.use_residual+1,1,1,1).reshape(-1,2)
        
        for i, meta in enumerate(metas):
            lines_pred = lines_batch[i].reshape(-1,4).detach()
            
            
            junction_gt = meta['junc']
            lines_gt = meta['lines']

            lines_matching_cost = torch.min(
                torch.sum((lines_pred[:,None]-lines_gt)**2,dim=-1),
                torch.sum((lines_pred[:,None]-lines_gt[:,[2,3,0,1]])**2,dim=-1),
            )
            mcost, mid = lines_matching_cost.min(dim=0)

            extra_info['recall_hafm-05'] += ((mcost<5).float().mean())
            extra_info['recall_hafm-10'] += ((mcost<10).float().mean())
            extra_info['recall_hafm-15'] += ((mcost<15).float().mean())

            lines_pred_labels = (lines_matching_cost.min(dim=1)[0]<15).float()
            
            
            lines_pred_feat = self.compute_loi_features(loi_features_aux[i],lines_pred.detach())
            lines_pred_logits = self.line_mlp(lines_pred_feat).flatten()
            
            loss_dict['loss_lineness'] += self.bce_loss(lines_pred_logits,lines_pred_labels.float()).mean()/batch_size
            
            N = junction_gt.size(0)   

            juncs_pred, _ = get_junctions(non_maximum_suppression(jloc_pred[i]),joff_pred[i], topk=min(N*2+2,self.n_dyn_junc))
            dis_junc_to_end1, idx_junc_to_end1 = torch.sum((lines_pred[:,:2]-juncs_pred[:,None])**2,dim=-1).min(0)
            dis_junc_to_end2, idx_junc_to_end2 = torch.sum((lines_pred[:, 2:] - juncs_pred[:, None]) ** 2, dim=-1).min(0)
            

            idx_junc_to_end_min = torch.min(idx_junc_to_end1,idx_junc_to_end2)
            idx_junc_to_end_max = torch.max(idx_junc_to_end1,idx_junc_to_end2)
            iskeep = idx_junc_to_end_min<idx_junc_to_end_max
            if self.j2l_threshold>0:
                iskeep *= (dis_junc_to_end1<self.j2l_threshold)*(dis_junc_to_end2<self.j2l_threshold)
            idx_lines_for_junctions = torch.stack((idx_junc_to_end_min[iskeep],idx_junc_to_end_max[iskeep]),dim=-1)
            idx_lines_for_junctions, inverse = torch.unique(idx_lines_for_junctions,sorted=True,return_inverse=True,dim=0)

            perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
            inverse, perm = inverse.flip([0]), perm.flip([0])
            perm = inverse.new_empty(idx_lines_for_junctions.size(0)).scatter_(0, inverse, perm)
            

            lines_init     = lines_pred[iskeep][perm]
            lines_adjusted = torch.cat((juncs_pred[idx_lines_for_junctions[:,0]], juncs_pred[idx_lines_for_junctions[:,1]]),dim=1)
            
            lines_matching_cost = torch.min(
                torch.sum((lines_adjusted[:,None]-lines_gt)**2,dim=-1),
                torch.sum((lines_adjusted[:,None]-lines_gt[:,[2,3,0,1]])**2,dim=-1),
            )
            mcost, mid = lines_matching_cost.min(dim=0)

            extra_info['recall_adjust-05'] += ((mcost<5).float().mean())
            extra_info['recall_adjust-10'] += ((mcost<10).float().mean())
            extra_info['recall_adjust-15'] += ((mcost<15).float().mean())
            
            cost_, match_ = torch.sum((juncs_pred-junction_gt[:,None])**2,dim=-1).min(0)
            
            match_[cost_>1.5*1.5] = N
            Lpos = meta['Lpos']
            Lneg = meta['Lneg']
            labels = Lpos[match_[idx_lines_for_junctions[:,0]],match_[idx_lines_for_junctions[:,1]]]
            

            lines_for_train = lines_adjusted
            lines_for_train_init = lines_init

            
            labels_for_train = labels

            lines_for_train = torch.cat((lines_for_train,meta['lpre']))
            lines_for_train_init = torch.cat((lines_for_train_init,meta['lpre']))
            labels_for_train = torch.cat((labels_for_train.float(),meta['lpre_label']))

            e1_features = self.bilinear_sampling(loi_features[i], lines_for_train[:,:2]-0.5).t()
            e2_features = self.bilinear_sampling(loi_features[i], lines_for_train[:,2:]-0.5).t()
            f1 = self.compute_loi_features(loi_features_thin[i],lines_for_train)
            f2 = self.compute_loi_features(loi_features_aux[i],lines_for_train_init)
            line_features = torch.cat((e1_features,e2_features,f1,f2),dim=-1)

            logits = self.fc2_head(self.fc2(line_features) + self.fc2_res(torch.cat((f1,f2),dim=-1)))

            if self.loi_cls_type == 'sigmoid':
                loss_ = self.loss(logits.flatten(), labels_for_train)
            else:
                loss_ = self.loss(logits, labels_for_train.long())
            
            if (labels_for_train==1).sum() == 0:
                loss_positive = torch.zeros_like(loss_[labels_for_train==1].mean())
            else:
                loss_positive = loss_[labels_for_train==1].mean()
            loss_negative = loss_[labels_for_train==0].mean()

            loss_dict['loss_pos'] += loss_positive/batch_size
            loss_dict['loss_neg'] += loss_negative/batch_size
        return loss_dict, extra_info

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
            'loss_aux': 0.0,
        }

        extra_info = defaultdict(list)

                
        loi_features = self.fc1(features)
        loi_features_thin = self.fc3(features)
        loi_features_aux = self.fc4(features)
        output = outputs[0]
        md_pred = output[:,:3].sigmoid()
        dis_pred = output[:,3:4].sigmoid()
        res_pred = output[:,4:5].sigmoid()
        jloc_pred= output[:,5:7].softmax(1)[:,1:]
        joff_pred= output[:,7:9].sigmoid() - 0.5
        

        batch_size = md_pred.size(0)

        lines_batch = self.hafm_decoding(md_pred, dis_pred, res_pred if self.use_residual else None, flatten=False, scale=self.hafm_encoder.dis_th)
        
        loss_dict_, extra_info = self.refinement_train(lines_batch, jloc_pred, joff_pred, loi_features, loi_features_thin,loi_features_aux, metas)

        
        loss_dict['loss_pos'] += loss_dict_['loss_pos']
        loss_dict['loss_neg'] += loss_dict_['loss_neg']
        loss_dict['loss_lineness'] = loss_dict_['loss_lineness']

        mask = targets['mask']

        lines_tgt = self.hafm_decoding(targets['md'], targets['dis'], None, flatten=False, scale=self.hafm_encoder.dis_th)

        mask2 = []
        for i in range(batch_size):
            lines_gt = metas[i]['lines']
            temp = lines_tgt[i].reshape(-1,4)
            temp_mask = torch.cdist(temp,lines_gt).min(dim=1)[0]<1.0
            temp_mask = temp_mask.reshape(lines_tgt[i].shape[:-1])
            mask2.append(temp_mask)
        mask2 = torch.stack(mask2)

        
        lines_tgt = lines_tgt.repeat((1,2*self.use_residual+1,1,1,1))
        lines_len = torch.sum((lines_tgt[...,:2]-lines_tgt[...,2:])**2,dim=-1)

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

                lines_learned = self.hafm_decoding(output[:,:3].sigmoid(), output[:,3:4].sigmoid(), output[:,4:5].sigmoid() if self.use_residual else None, flatten=False, scale=self.hafm_encoder.dis_th)
                
                wt = 1/lines_len.clamp_min(1.0)*mask2
                loss_map = F.l1_loss(lines_learned, lines_tgt,reduction='none').mean(dim=-1)
                
                loss_dict['loss_aux'] += torch.mean(loss_map*wt)/torch.mean(mask)
                
        for key in extra_info.keys():
            extra_info[key] = extra_info[key]/batch_size

        return loss_dict, extra_info

    def hafm_decoding_mask(self, md_maps, dis_maps, residual_maps, scores, scale=5.0):
        device = md_maps.device

        batch_size, _, height, width = md_maps.shape
        _y = torch.arange(0,height,device=device).float()
        _x = torch.arange(0,width, device=device).float()

        y0, x0 =torch.meshgrid(_y, _x,indexing='ij')
        y0 = y0.reshape(1,1,-1)
        x0 = x0.reshape(1,1,-1)
        
        sign_pad = torch.arange(-self.use_residual,self.use_residual+1,device=device,dtype=torch.float32).reshape(1,-1,1)

        if residual_maps is not None:
            residual = residual_maps.reshape(batch_size,1,-1)*sign_pad
            distance_fields = dis_maps.reshape(batch_size,1,-1) + residual
            scores = scores.reshape(batch_size,1,-1).repeat((1,2*self.use_residual+1,1))
        else:
            distance_fields = dis_maps.reshape(batch_size,1,-1)
            scores = scores.reshape(batch_size,1,-1)
        md_maps = md_maps.reshape(batch_size,3,-1)
        
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

        lines = lines.reshape(batch_size,-1,4)
        scores = scores.reshape(batch_size,-1)
        
        sc_, arg_ = scores[0].sort(descending=True)
        lines_out = lines[0][arg_[sc_>0]]
        
        return lines_out, sc_[sc_>0]

    def hafm_decoding(self, md_maps, dis_maps, residual_maps, scale=5.0, flatten = True):
        device = md_maps.device

        batch_size, _, height, width = md_maps.shape
        _y = torch.arange(0,height,device=device).float()
        _x = torch.arange(0,width, device=device).float()

        y0, x0 =torch.meshgrid(_y, _x,indexing='ij')
        y0 = y0[None,None]
        x0 = x0[None,None]
        
        sign_pad = torch.arange(-self.use_residual,self.use_residual+1,device=device,dtype=torch.float32).reshape(1,-1,1,1)

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

        
