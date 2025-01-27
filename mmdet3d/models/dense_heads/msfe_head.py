import copy

import torch
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from torch import nn

from ..builder import HEADS, build_loss
from torch.cuda.amp.autocast_mode import autocast

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y

def _neg_loss(pred, gt):
    pos_inds = gt.gt(0).float()
    neg_inds = gt.eq(0).float()

    pos_weights = torch.pow(gt, 2)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_weights * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

    # num_pos  = pos_inds.float().sum()
    num_pos  = (pos_weights * pos_inds).sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class FocalLoss(nn.Module):
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)

class RegL1Loss(nn.Module):
  def __init__(self):
    super(RegL1Loss, self).__init__()
  
  def forward(self, output, target, mask):
    mask = mask.unsqueeze(2).expand_as(output).float()
    # loss = nn.functional.l1_loss(output * mask, target * mask, size_average=False)
    loss = nn.functional.l1_loss(output * mask, target * mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss


@HEADS.register_module()
class MSFEHead(BaseModule):
    def __init__(self, 
                 source_scale_idx,
                 target_scale_idx,
                 num_topk=100, 
                 num_per_topk=100,
                 num_classes=10,
                 in_channel=512,
                 with_reg=True,
                #  loss_hms=None,
                 **kwargs):
        super(MSFEHead, self).__init__(**kwargs)
        self.source_scale_idx = source_scale_idx
        self.target_scale_idx = target_scale_idx
        self.downsample_ratio = 2 ** (self.target_scale_idx - self.source_scale_idx)
        # self.num_topk = num_topk
        # self.num_per_topk = num_per_topk
        # self.with_reg = with_reg
        self.num_classes = num_classes

        # self.loss_hm = build_loss(loss_hms)
        self.loss_hm = FocalLoss()
        self.hm_conv = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel,
                        kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channel, num_classes, 
                        kernel_size=1, stride=1, 
                        padding=0, bias=True))
        self.hm_conv[-1].bias.data.fill_(-2.19)

        # if with_reg:
        #     self.loss_reg = RegL1Loss()
        #     self.reg_conv = nn.Sequential(
        #             nn.Conv2d(in_channel, in_channel,
        #                 kernel_size=3, padding=1, bias=True),
        #             nn.ReLU(inplace=True),
        #             nn.Conv2d(in_channel, 2, 
        #                 kernel_size=1, stride=1, 
        #                 padding=0, bias=True))
        #     fill_fc_weights(self.reg_conv)
            
     

    @autocast(enabled=False)
    def forward(self, inputs): 
        x = inputs[self.source_scale_idx]
        x = x.float()
        
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)

        preds_hms = self.hm_conv(x).view(B, N, self.num_classes, H, W)
        # if self.with_reg:
        #     preds_regs = self.reg_conv(x).view(B, N, 2, H, W)
        
        return preds_hms

    @force_fp32()
    def get_loss(self, preds_hms, gt_hms):
        loss_dict = dict()
        preds_hms = _sigmoid(preds_hms)
        loss_dict['loss_hm'] = self.loss_hm(preds_hms, gt_hms)
        
        # if self.with_reg:
        #     loss_dict['loss_reg'] = self.loss_reg(preds_regs, gt_regs, gt_regs_mask)

        return loss_dict
    
    # def get_aug_feature1(self, hms, muti_scale_feature):
    #     target_x = muti_scale_feature[self.target_scale_idx].clone()
    #     B_t, N_t, C_t, H_t, W_t = target_x.shape
       
    #     # 多尺度采样
    #     hms_s = torch.sigmoid(hms)
    #     hms_s = torch.max(hms_s, 2, keepdim=True)[0]
    #     hms_s = torch.where(hms_s > 0.1, hms_s, torch.zeros_like(hms_s))

    #     f_0 = muti_scale_feature[0]
    #     f_1 = muti_scale_feature[1]

    #     # f_0
    #     B, N, C, H, W = f_0.shape
    #     f_0 = f_0.view(B * N, C, H, W)
    #     f_0 = nn.functional.interpolate(f_0, 
    #                                     size=(H//2, W//2), 
    #                                     mode='bilinear', 
    #                                     align_corners=True)
    #     f_0 = f_0.view(B, N, C, H//2, W//2)
    #     f_0_seg = hms_s * f_0
    #     f_0_seg = f_0_seg.view(B * N, C, H//2, W//2)
    #     f_0_seg = nn.functional.interpolate(f_0_seg, 
    #                                         size=(H//4, W//4), 
    #                                         mode='bilinear', 
    #                                         align_corners=True)
    #     f_0_seg = f_0_seg.view(B, N, C, H//4, W//4)
    #     target_x += f_0_seg

    #     # f_1
    #     B, N, C, H, W = f_1.shape
    #     f_1_seg = hms_s * f_1
    #     f_1_seg = f_1_seg.view(B * N, C, H, W)
    #     f_1_seg = nn.functional.interpolate(f_1_seg, 
    #                                         size=(H//2, W//2), 
    #                                         mode='bilinear', 
    #                                         align_corners=True)
    #     f_1_seg = f_1_seg.view(B, N, C, H//2, W//2)
    #     target_x += f_1_seg

    #     # hm
    #     hms_s = hms_s.view(B*N, 1, H, W)
    #     hms_s = nn.functional.max_pool2d(hms_s, (2, 2), stride=2, padding=0)
    #     hms_s = hms_s.view(B, N, 1, H//2, W//2)

    #     return target_x, hms_s

    def get_aug_feature(self, hms, muti_scale_feature):
        target_x = muti_scale_feature[self.target_scale_idx].clone()
        B_t, N_t, C_t, H_t, W_t = target_x.shape
        
        # multi-scale sampling
        hms_s = torch.sigmoid(hms)
        hms_s = torch.max(hms_s, 2, keepdim=True)[0]
        hms_s = torch.where(hms_s > 0.1, hms_s, torch.zeros_like(hms_s))
        for i in range(self.target_scale_idx):
            tmp_feature = muti_scale_feature[i]
            tmp_feature_seg = hms_s * tmp_feature

            B, N, C, H, W = tmp_feature_seg.shape
            tmp_feature_seg = tmp_feature_seg.view(B * N, C, H, W)
            tmp_feature_seg = nn.functional.interpolate(tmp_feature_seg, 
                                                        size=(H_t, W_t), 
                                                        mode='bilinear', 
                                                        align_corners=True)
            tmp_feature_seg = tmp_feature_seg.view(B, N, C, H_t, W_t)
            target_x += tmp_feature_seg

            hms_s = hms_s.view(B*N, 1, H, W)
            hms_s = nn.functional.max_pool2d(hms_s, (2, 2), stride=2, padding=0)
            hms_s = hms_s.view(B, N, 1, H//2, W//2)

        return target_x, hms_s

    def get_aug_feature_v2(self, hms, muti_scale_feature):
        target_x = muti_scale_feature[self.target_scale_idx].clone()
        B_t, N_t, C_t, H_t, W_t = target_x.shape
       
        hms_s = torch.sigmoid(hms)
        hms_s = torch.max(hms_s, 2, keepdim=True)[0]
        hms_s = torch.where(hms_s > 0.1, hms_s, torch.zeros_like(hms_s))

        # f_0
        f_0 = muti_scale_feature[0]
        f_0_seg = hms_s * f_0
        B, N, C, H, W = f_0_seg.shape
        f_0_seg = f_0_seg.view(B * N, C, H, W)
        f_0_seg = nn.functional.interpolate(f_0_seg, 
                                            size=(H_t, W_t), 
                                            mode='bilinear', 
                                            align_corners=True)
        f_0_seg = f_0_seg.view(B, N, C, H_t, W_t)
        target_x += f_0_seg

        # f_1
        hms_s = hms_s.view(B*N, 1, H, W)
        hms_s = nn.functional.max_pool2d(hms_s, (2, 2), stride=2, padding=0)
        hms_s = hms_s.view(B, N, 1, H//2, W//2)
        f_1 = muti_scale_feature[1]
        f_1_seg = hms_s * f_1
        B, N, C, H, W = f_1_seg.shape
        f_1_seg = f_1_seg.view(B * N, C, H, W)
        f_1_seg = nn.functional.interpolate(f_1_seg, 
                                            size=(H_t, W_t), 
                                            mode='bilinear', 
                                            align_corners=True)
        f_1_seg = f_1_seg.view(B, N, C, H_t, W_t)
        target_x += f_1_seg

        return target_x

    # def get_aug_feature(self, hms_regs, muti_scale_feature):
    #     # return muti_scale_feature[self.target_scale_idx].clone()
    #     hms, regs = hms_regs
    #     B, N, C, H, W = hms.size()

    #     topk_score, topk_views, topk_pos = self.ct_topk(hms, regs)

    #     topk_pos_center = topk_pos.clone()
    #     topk_pos_center = (topk_pos_center - 0.5) * 2

    #     pos_list = []
    #     ref_points = torch.full((B, N, self.num_topk, 2), -2, dtype=torch.float32).to(topk_pos_center) # 每个视图中最多只有num_topk个，初始设置位置比例为-2
    #     for num_cam in range(N):
    #         pos = torch.where(topk_views == num_cam)
    #         if len(pos[0]) != 0:
    #             ref_points[pos[0], num_cam, pos[1], :] = topk_pos_center[pos[0], pos[1], :]
    #         pos_list.append(pos)
        
    #     sampled_feats = []
    #     useful_muti_scale_feature = muti_scale_feature[:self.target_scale_idx]
    #     for lvl, feature in enumerate(useful_muti_scale_feature):
    #         b, n, c, h, w = feature.size()
    #         feature = feature.view(b * n, c, h, w)
    #         ref_points_lvl = ref_points.view(b * n, self.num_topk, 1, 2)
    #         sampled_feat = nn.functional.grid_sample(feature, ref_points_lvl, align_corners=True)
    #         sampled_feats.append(sampled_feat)
    #     sampled_feats = torch.stack(sampled_feats, -1)
    #     sampled_feats = sampled_feats * (self.l_weight.sigmoid())
    #     sampled_feats = sampled_feats.sum(-1)
    #     sampled_feats = sampled_feats.view(B, N, -1, self.num_topk)

    #     target_x = muti_scale_feature[self.target_scale_idx].clone()   # clone is important
    #     b_t, n_t, c_t, h_t, w_t = target_x.size()
    #     b_s, n_s, c_s, _ = sampled_feats.size()
    #     assert (b_t == b_s) and (n_t == n_s) and (c_t == c_s), 'Dimesion doesnt match'

    #     for num_cam, pos in enumerate(pos_list):
    #         if len(pos[0]) == 0:
    #             continue 

    #         pos_mask = (topk_score[pos[0], pos[1]] > 0.1).unsqueeze(-1)
    #         sampled_feats[pos[0], num_cam, :, pos[1]] *= pos_mask
    #         # pos_filter = torch.where(topk_score[pos[0], pos[1]] < 0.3)[0]
    #         # sampled_feats[pos[0][pos_filter], num_cam, :, pos[1][pos_filter]] = 0

    #         pos_in_targetx = topk_pos[pos[0], pos[1]]
    #         pos_in_targetx[:, 0] = pos_in_targetx[:, 0] * w_t
    #         pos_in_targetx[:, 1] = pos_in_targetx[:, 1] * h_t
    #         pos_in_targetx = pos_in_targetx.long()

    #         target_x[pos[0], num_cam, :, pos_in_targetx[:,1], pos_in_targetx[:,0]] += sampled_feats[pos[0], num_cam, :, pos[1]]

    #     return target_x
    
    # def ct_topk(self, heat, reg):
    #     B, N, C, H, W = heat.size()
    #     heat = heat.view(B, N * C, H, W)

    #     heat = torch.sigmoid(heat)
    #     # perform nms on heatmaps
    #     heat = _nms(heat)
        
    #     topk_score, clses, ys, xs = self._topk(heat)
        
    #     topk_views = (clses / C).int()
    #     # topk_clses = (clses % C).int()
        
    #     # add res
    #     if self.with_reg:
    #         # reg_d = reg.detach()
    #         reg_d = reg.gather(1, topk_views.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(B, self.num_topk, 2, H, W).long())
    #         reg_d = reg_d.gather(3, ys.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(B, self.num_topk, 2, 1, W).long())
    #         reg_d = reg_d.gather(4, xs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(B, self.num_topk, 2, 1, 1).long()).squeeze(-1).squeeze(-1)
    #         reg_d = torch.clamp(reg_d, 0 , 1 - 1e-4)
    #         topk_pos = torch.stack((xs, ys), -1) + reg_d
    #     else:
    #         topk_pos = torch.stack((xs, ys), -1) + 0.5

    #     topk_pos[..., 0] /= W 
    #     topk_pos[..., 1] /= H 


    #     return topk_score, topk_views, topk_pos

    
    # def _topk(self, scores):
    #     batch, cat, height, width = scores.size()
        
    #     topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), self.num_per_topk)

    #     topk_inds = topk_inds % (height * width)
    #     topk_ys   = (topk_inds / width).int().float()
    #     topk_xs   = (topk_inds % width).int().float()
        
    #     topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), self.num_topk)
    #     topk_clses = (topk_ind / self.num_per_topk).int()

    #     # topk_inds = _gather_feat(
    #         # topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    #     topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, self.num_topk)
    #     topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, self.num_topk)

    #     return topk_score, topk_clses, topk_ys, topk_xs  


