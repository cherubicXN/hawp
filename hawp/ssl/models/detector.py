from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import time

from hawp.fsl.backbones import build_backbone
from .hafm import HAFMencoder
from .base import HAWPBase
from .losses import *
from .heatmap_decoder import PixelShuffleDecoder

from .registry import MODELS

@MODELS.register('HAWP')
class HAWP(HAWPBase):
    def __init__(self, cfg, *, gray_scale=False):
        super(HAWP, self).__init__(
            num_points = cfg.MODEL.LOI_POOLING.NUM_POINTS,
            num_residuals = int(cfg.MODEL.PARSING_HEAD.USE_RESIDUAL),
            distance_threshold =  cfg.ENCODER.DIS_TH)

        self.hafm_encoder = HAFMencoder(cfg)

        self.backbone = build_backbone(cfg,gray_scale=gray_scale)

        self.n_dyn_junc = cfg.MODEL.PARSING_HEAD.N_DYN_JUNC
        self.n_dyn_posl = cfg.MODEL.PARSING_HEAD.N_DYN_POSL
        self.n_dyn_negl = cfg.MODEL.PARSING_HEAD.N_DYN_NEGL
        self.n_dyn_othr = cfg.MODEL.PARSING_HEAD.N_DYN_OTHR
        self.n_dyn_othr2= cfg.MODEL.PARSING_HEAD.N_DYN_OTHR2

        #Matcher
        self.j2l_threshold = cfg.MODEL.PARSING_HEAD.J2L_THRESHOLD
        self.jmatch_threshold = cfg.MODEL.PARSING_HEAD.JMATCH_THRESHOLD

        # LOI POOLING
        self.use_init_lines = cfg.MODEL.LOI_POOLING.USE_INIT_LINES
        self.dim_junction_feature    = cfg.MODEL.LOI_POOLING.DIM_JUNCTION_FEATURE
        self.dim_junction_feature    = cfg.MODEL.LOI_POOLING.DIM_JUNCTION_FEATURE
        self.dim_edge_feature = cfg.MODEL.LOI_POOLING.DIM_EDGE_FEATURE
        self.dim_fc     = cfg.MODEL.LOI_POOLING.DIM_FC


        self.n_out_junc = cfg.MODEL.PARSING_HEAD.N_OUT_JUNC
        self.n_out_line = cfg.MODEL.PARSING_HEAD.N_OUT_LINE

        # TODO: add to cfg
        self.num_junctions_inference = 300
        self.junction_threshold_hm = 0.008
        self.use_residual = int(cfg.MODEL.PARSING_HEAD.USE_RESIDUAL)
        
        assert cfg.MODEL.LOI_POOLING.TYPE in ['softmax', 'sigmoid']
        assert cfg.MODEL.LOI_POOLING.ACTIVATION in ['relu', 'gelu']

        self.loi_cls_type = cfg.MODEL.LOI_POOLING.TYPE
        self.loi_layer_norm = cfg.MODEL.LOI_POOLING.LAYER_NORM
        self.loi_activation = nn.ReLU if cfg.MODEL.LOI_POOLING.ACTIVATION == 'relu' else nn.GELU        

        self.fc1 = nn.Conv2d(256, self.dim_junction_feature, 1)

        self.fc3 = nn.Conv2d(256, self.dim_edge_feature, 1)
        self.fc4 = nn.Conv2d(256, self.dim_edge_feature, 1)

        self.regional_head = nn.Conv2d(256, 1, 1)
        fc2 = [nn.Linear(self.dim_junction_feature*2 + (self.num_points-2)*self.dim_edge_feature*(1+self.use_init_lines), self.dim_fc),
        ]
        for i in range(2):
            fc2.append(nn.ReLU(True))
            fc2.append(nn.Linear(self.dim_fc,self.dim_fc))

        
        self.fc2 = nn.Sequential(*fc2)
        self.fc2_res = nn.Sequential(nn.Linear((1+self.use_init_lines)*(self.num_points-2)*self.dim_edge_feature, self.dim_fc),nn.ReLU(True))

        self.line_mlp = nn.Sequential(
            nn.Linear((self.num_points-2)*self.dim_edge_feature,128),
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
        
        self.use_heatmap_decoder = cfg.MODEL.USE_LINE_HEATMAP
        if self.use_heatmap_decoder:
            self.heatmap_decoder = PixelShuffleDecoder(input_feat_dim=256)

        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.train_step = 0

    
    def wireframe_matcher(self, juncs_pred, lines_pred, is_train=False, is_shuffle=False):
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


        if is_shuffle:
            cost_atoi_argsort = torch.randperm(iskeep.sum(),device=juncs_pred.device)
        else:
            cost_atoi = torch.min(
             torch.sum((juncs_pred[idx_lines_for_junctions].reshape(-1,4) - lines_pred[iskeep])**2,dim=-1),
             torch.sum((juncs_pred[idx_lines_for_junctions].reshape(-1,4) - lines_pred[iskeep][:,[2,3,0,1]])**2,dim=-1)
             )

            cost_atoi_argsort = cost_atoi.argsort(descending=True)
            

        lines_pred_kept = lines_pred[iskeep][cost_atoi_argsort]
        idx_lines_for_junctions = idx_lines_for_junctions[cost_atoi_argsort]

        _, perm = np.unique(idx_lines_for_junctions.cpu().numpy(),return_index=True,axis=0)

        idx_lines_for_junctions = idx_lines_for_junctions[perm]
        lines_init = lines_pred_kept[perm]
        
        if is_train:
            idx_lines_for_junctions_mirror = torch.cat((idx_lines_for_junctions[:,1,None],idx_lines_for_junctions[:,0,None]),dim=1)
            idx_lines_for_junctions = torch.cat((idx_lines_for_junctions, idx_lines_for_junctions_mirror))
        
        #lines_adjusted = torch.cat((juncs_pred[idx_lines_for_junctions[:,0]], juncs_pred[idx_lines_for_junctions[:,1]]),dim=1)
        lines_adjusted = juncs_pred[idx_lines_for_junctions].reshape(-1,4)
        #lines_adjusted = torch.stack((juncs_pred[idx_lines_for_junctions[:,0]], juncs_pred[idx_lines_for_junctions[:,1]]),dim=1)
        
        return lines_adjusted, lines_init, perm, idx_lines_for_junctions

    def forward_test_with_junction(self, images, junctions, annotations = None):
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
        
        lines_pred = self.hafm_decoding(md_pred, dis_pred, res_pred if self.use_residual else None, flatten = True)[0]

        
        juncs_pred = junctions
        
        _ = torch.ones((juncs_pred.shape[0]),dtype=juncs_pred.dtype,device=device)
        

        
        lines_adjusted, lines_init, perm, _ = self.wireframe_matcher(juncs_pred, lines_pred)
        
        e1_features = self.bilinear_sampling(loi_features[0], lines_adjusted[:,:2]-0.5).t()
        e2_features = self.bilinear_sampling(loi_features[0], lines_adjusted[:,2:]-0.5).t()

        f1 = self.compute_loi_features(loi_features_thin[0],lines_adjusted, tspan=self.tspan[...,1:-1])
        line_features = torch.cat((e1_features,e2_features,f1),dim=-1)

        if self.use_init_lines:
            f2 = self.compute_loi_features(loi_features_aux[0],lines_init, tspan=self.tspan[...,1:-1])
            line_features = torch.cat((line_features,f2),dim=-1)
            logits = self.fc2_head(self.fc2(line_features)+self.fc2_res(torch.cat((f1,f2),dim=-1)))
        else:
            logits = self.fc2_head(self.fc2(line_features)+self.fc2_res(f1))

        if self.loi_cls_type == 'softmax':
            scores = logits.softmax(dim=-1)[:,1]
        else:
            scores = logits.sigmoid()[:,0]
        
        sarg = torch.argsort(scores,descending=True)

        lines_final = lines_adjusted[sarg]
        score_final = scores[sarg]
        lines_before = lines_init[sarg]

        num_detection = min((score_final>0.00).sum(),1000)
        lines_final = lines_final[:num_detection]
        score_final = score_final[:num_detection]

        juncs_final = juncs_pred
        juncs_score = _

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
        
        lines_pred = self.hafm_decoding(md_pred, dis_pred, res_pred if self.use_residual else None, flatten = True)[0]


        jloc_pred_nms = self.non_maximum_suppression(jloc_pred[0])

        #building
        # self.num_junctions_inference = 512
        topK = min(self.num_junctions_inference, int((jloc_pred_nms>self.junction_threshold_hm).float().sum().item()))
        
        juncs_pred, _ = self.get_junctions(jloc_pred_nms,joff_pred[0], topk=topK,th=self.junction_threshold_hm)
        if juncs_pred.shape[0] == 0:
            return {'lines_pred': None, 'lines_score': None}
        lines_adjusted, lines_init, perm, _ = self.wireframe_matcher(juncs_pred, lines_pred)
        matcher_time = time.time()
        # lines_adjusted_trials = []
        # lines_init_trials = []
        # for i in range(10):
        #     lines_adjusted, lines_init, perm = self.wireframe_matcher(juncs_pred, lines_pred)
        #     lines_adjusted_trials.append(lines_adjusted)
        #     lines_init_trials.append(lines_init)
        # matcher_time = time.time()-matcher_time
        # lines_init_trials = torch.stack(lines_init_trials)
        # is_stable = lines_init_trials.std(dim=0).mean(dim=-1)<1e-6
        
        e1_features = self.bilinear_sampling(loi_features[0], lines_adjusted[:,:2]-0.5).t()
        e2_features = self.bilinear_sampling(loi_features[0], lines_adjusted[:,2:]-0.5).t()

        f1 = self.compute_loi_features(loi_features_thin[0],lines_adjusted, tspan=self.tspan[...,1:-1])
        if self.use_init_lines:
            f2 = self.compute_loi_features(loi_features_aux[0],lines_init, tspan=self.tspan[...,1:-1])
            line_features = torch.cat((e1_features,e2_features,f1,f2),dim=-1)
            logits = self.fc2_head(self.fc2(line_features)+self.fc2_res(torch.cat((f1,f2),dim=-1)))
        else:
            line_features = torch.cat((e1_features,e2_features,f1),dim=-1)
            logits = self.fc2_head(self.fc2(line_features)+self.fc2_res(f1))


        if self.loi_cls_type == 'softmax':
            scores = logits.softmax(dim=-1)[:,1]
        else:
            scores = logits.sigmoid()[:,0]

        sarg = torch.argsort(scores,descending=True)

        lines_final = lines_adjusted[sarg]
        score_final = scores[sarg]
        lines_before = lines_init[sarg]

        num_detection = min((score_final>0.00).sum(),1000)
        lines_final = lines_final[:num_detection]
        score_final = score_final[:num_detection]

        juncs_final = juncs_pred
        juncs_score = _

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

    def forward(self, images, annotations = None, targets = None):
        if self.training:
            return self.forward_train(images, annotations=annotations)
        else:
            return self.forward_test(images, annotations=annotations)

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
        valid_mask = annotations['valid_mask']
        if self.use_heatmap_decoder:
            heatmaps_pred = self.heatmap_decoder(features)
            loss_dict['loss_heatmap'] = torch.mean((self.loss(heatmaps_pred,annotations['heatmap'][:,0].long()))*valid_mask)
        extra_info = defaultdict(list)

                
        loi_features = self.fc1(features)
        loi_features_thin = self.fc3(features)
        loi_features_aux = self.fc4(features)
        # regional_scores = self.regional_head(features)
        output = outputs[0]
        md_pred = output[:,:3].sigmoid()
        dis_pred = output[:,3:4].sigmoid()
        res_pred = output[:,4:5].sigmoid()
        jloc_pred= output[:,5:7].softmax(1)[:,1:]
        joff_pred= output[:,7:9].sigmoid() - 0.5
        
        # regional_scores = 

        # lines_batch = []

        batch_size = md_pred.size(0)

        lines_batch = self.hafm_decoding(md_pred, dis_pred, res_pred if self.use_residual else None, flatten=False, scale=self.hafm_encoder.dis_th)
        
        loss_dict_, extra_info = self.refinement_train(lines_batch, jloc_pred, joff_pred, loi_features, loi_features_thin,loi_features_aux, metas)

        
        loss_dict['loss_pos'] += loss_dict_['loss_pos']
        loss_dict['loss_neg'] += loss_dict_['loss_neg']
        loss_dict['loss_lineness'] = loss_dict_['loss_lineness']

        mask = targets['mask']

        lines_tgt = self.hafm_decoding(targets['md'], targets['dis'], None, flatten=False, scale=self.hafm_encoder.dis_th)

        mask2 = torch.zeros_like(targets['dis'],dtype=torch.bool)
        for i in range(batch_size):
            if metas[i]['lines'].shape[0]>0:
                lines_gt = metas[i]['lines']
                temp = lines_tgt[i].reshape(-1,4)
                temp_mask = torch.cdist(temp,lines_gt).min(dim=1)[0]<1.0
                temp_mask = temp_mask.reshape(lines_tgt[i].shape[:-1])
                mask2[i] = temp_mask
        # mask2 = torch.stack(mask2)

        # mask = mask2.float()
        # epsilon = (mask.sum().item()==0)*1e-6
        epsilon = 0
        
        lines_tgt = lines_tgt.repeat((1,2*self.use_residual+1,1,1,1))
        lines_len = torch.sum((lines_tgt[...,:2]-lines_tgt[...,2:])**2,dim=-1)

        if targets is not None:
            for nstack, output in enumerate(outputs):
                loss_map = torch.mean(F.l1_loss(output[:,:3].sigmoid(), targets['md'],reduction='none'),dim=1,keepdim=True)
                
                loss_dict['loss_md']  += torch.mean(loss_map*mask) / (torch.mean(mask)+epsilon)
                loss_map = F.l1_loss(output[:,3:4].sigmoid(), targets['dis'], reduction='none')
                loss_dict['loss_dis'] += torch.mean(loss_map*mask) /(torch.mean(mask)+epsilon)
                loss_residual_map = F.l1_loss(output[:,4:5].sigmoid(), loss_map, reduction='none')
                loss_dict['loss_res'] += torch.mean(loss_residual_map*mask)/(torch.mean(mask)+epsilon)
                loss_dict['loss_jloc'] += cross_entropy_loss_for_junction(output[:,5:7], targets['jloc'])
                loss_dict['loss_joff'] += sigmoid_l1_loss(output[:,7:9], targets['joff'], -0.5, targets['jloc'])

                lines_learned = self.hafm_decoding(output[:,:3].sigmoid(), output[:,3:4].sigmoid(), output[:,4:5].sigmoid() if self.use_residual else None, flatten=False, scale=self.hafm_encoder.dis_th)
                
                wt = 1/lines_len.clamp_min(1.0)*mask2
                loss_map = F.l1_loss(lines_learned, lines_tgt,reduction='none').mean(dim=-1)
                
                loss_dict['loss_aux'] += torch.mean(loss_map*wt)/(torch.mean(mask)+epsilon)

        for key in extra_info.keys():
            extra_info[key] = extra_info[key]/batch_size

        return loss_dict, extra_info

    def refinement_train(self, lines_batch, jloc_pred, joff_pred, loi_features, loi_features_thin, loi_features_aux, metas):
        loss_dict = defaultdict(float)
        extra_info = defaultdict(float)
        batch_size = lines_batch.shape[0]
        device = lines_batch.device

        
        for i, meta in enumerate(metas):
            if meta['lines'].shape[0] == 0:# or meta['junc'].shape[0] == 0:
                continue
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
            

            # valid_idx = (lines_matching_cost.min(dim=1)[0]<100)
            
            lines_pred_feat = self.compute_loi_features(loi_features_aux[i],lines_pred.detach(),self.tspan[...,1:-1])
            lines_pred_logits = self.line_mlp(lines_pred_feat).flatten()
            # lines_pred_labels = lines_pred_labels[valid_idx]
            
            loss_dict['loss_lineness'] += self.bce_loss(lines_pred_logits,lines_pred_labels.float()).mean()/batch_size
            
            N = junction_gt.size(0)   

            with torch.no_grad():
                juncs_pred, _ = self.get_junctions(self.non_maximum_suppression(jloc_pred[i]),joff_pred[i], topk=min(N*2+2,self.n_dyn_junc))
            if juncs_pred.shape[0] == 0:
                continue
            lines_adjusted, lines_init, _, idx_lines_for_junctions = self.wireframe_matcher(juncs_pred, lines_pred.detach(),is_shuffle=True)
            if lines_adjusted.shape[0] == 0:
                continue
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
            labels = Lpos[match_[idx_lines_for_junctions[:,0]],match_[idx_lines_for_junctions[:,1]]]

            lines_for_train = lines_adjusted
            lines_for_train_init = lines_init
            # import matplotlib.pyplot as plt
            # for line_a, line_i in zip(lines_for_train.detach().cpu().numpy()*4,lines_for_train_init.detach().cpu().numpy()*4):
            #     plt.plot([line_a[0],line_a[2]],
            #             [line_a[1],line_a[3]],'r-'
            #     )
            #     plt.plot([line_i[0],line_i[2]],
            #             [line_i[1],line_i[3]],'b-'
            #     )
            # plt.show()

            
            labels_for_train = labels

            lines_for_train = torch.cat((lines_for_train,meta['lpre']))
            lines_for_train_init = torch.cat((lines_for_train_init,meta['lpre']))
            labels_for_train = torch.cat((labels_for_train.float(),meta['lpre_label']))

            e1_features = self.bilinear_sampling(loi_features[i], lines_for_train[:,:2]-0.5).t()
            e2_features = self.bilinear_sampling(loi_features[i], lines_for_train[:,2:]-0.5).t()
            f1 = self.compute_loi_features(loi_features_thin[i],lines_for_train,self.tspan[...,1:-1])
            line_features = torch.cat((e1_features,e2_features,f1),dim=-1)
            
            if self.use_init_lines:
                f2 = self.compute_loi_features(loi_features_aux[i],lines_for_train_init,self.tspan[...,1:-1])
                line_features = torch.cat((line_features, f2),dim=-1)
                logits = self.fc2_head(self.fc2(line_features) + self.fc2_res(torch.cat((f1,f2),dim=-1)))
            else:
                logits = self.fc2_head(self.fc2(line_features) + self.fc2_res(f1))


            if self.loi_cls_type == 'sigmoid':
                loss_ = self.loss(logits.flatten(), labels_for_train)
            else:
                loss_ = self.loss(logits, labels_for_train.long())
            
            if (labels_for_train==1).sum() == 0:
                loss_positive = torch.zeros_like(loss_[labels_for_train==1].mean())
            else:
                loss_positive = loss_[labels_for_train==1].mean()

            if (labels_for_train==0).sum() == 0:
                loss_negative = torch.zeros_like(loss_[labels_for_train==0].mean())
            else:
                loss_negative = loss_[labels_for_train==0].mean()

            loss_dict['loss_pos'] += loss_positive/batch_size
            loss_dict['loss_neg'] += loss_negative/batch_size
        return loss_dict, extra_info