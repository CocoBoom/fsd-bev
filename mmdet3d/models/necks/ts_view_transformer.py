import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, force_fp32

from mmdet3d.ops.bev_pool_v2.bev_pool import bev_pool_v2
from ..builder import NECKS
from .view_transformer import LSSViewTransformerBEVDepth, DepthNet
from torch.cuda.amp.autocast_mode import autocast


@NECKS.register_module()
class TSViewTransformer(LSSViewTransformerBEVDepth):
    def __init__(self, depth_threshold=1, semantic_threshold=0.1, **kwargs):
        super(TSViewTransformer, self).__init__(**kwargs)
        self.depth_threshold = depth_threshold / self.D
        self.semantic_threshold = semantic_threshold
    
    def get_downsampled_gt_depth_and_semantic(self, gt_depths, gt_semantics):
        # remove point not in depth range
        gt_semantics[gt_depths < self.grid_config['depth'][0]] = 0
        gt_semantics[gt_depths > self.grid_config['depth'][1]] = 0
        gt_depths[gt_depths < self.grid_config['depth'][0]] = 0
        gt_depths[gt_depths > self.grid_config['depth'][1]] = 0
        gt_semantic_depths = gt_depths * gt_semantics

        B, N, H, W = gt_semantics.shape
        gt_semantics = gt_semantics.view(
            B * N,
            H // self.downsample,
            self.downsample,
            W // self.downsample,
            self.downsample,
            1,
        )
        gt_semantics = gt_semantics.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_semantics = gt_semantics.view(
            -1, self.downsample * self.downsample)
        gt_semantics = torch.max(gt_semantics, dim=-1).values
        gt_semantics = gt_semantics.view(B * N, H // self.downsample,
                                   W // self.downsample)
        gt_semantics = F.one_hot(gt_semantics.long(),
                              num_classes=2).view(-1, 2).float()

        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample,
            self.downsample,
            W // self.downsample,
            self.downsample,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   W // self.downsample)
        gt_depths = (gt_depths -
                     (self.grid_config['depth'][0] - self.grid_config['depth'][2])) / self.grid_config['depth'][2]
        gt_depths = torch.where(
            (gt_depths < self.D + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.D + 1).view(
                                  -1, self.D + 1)[:, 1:].float()
        gt_semantic_depths = gt_semantic_depths.view(
            B * N,
            H // self.downsample,
            self.downsample,
            W // self.downsample,
            self.downsample,
            1,
        )
        gt_semantic_depths = gt_semantic_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_semantic_depths = gt_semantic_depths.view(
            -1, self.downsample * self.downsample)
        gt_semantic_depths =  torch.where(gt_semantic_depths == 0.0,
                                    1e5 * torch.ones_like(gt_semantic_depths),
                                    gt_semantic_depths)
        gt_semantic_depths = (gt_semantic_depths - (self.grid_config['depth'][0] - 
                            self.grid_config['depth'][2])) / self.grid_config['depth'][2] 
        gt_semantic_depths = torch.where(
                    (gt_semantic_depths < self.D + 1) & (gt_semantic_depths >= 0.0),
                    gt_semantic_depths, torch.zeros_like(gt_semantic_depths)).long()                           
        soft_depth_mask = gt_semantics[:,1] > 0
        gt_semantic_depths = gt_semantic_depths[soft_depth_mask]
        gt_semantic_depths_cnt = gt_semantic_depths.new_zeros([gt_semantic_depths.shape[0], self.D+1])
        for i in range(self.D+1):
            gt_semantic_depths_cnt[:,i] = (gt_semantic_depths == i).sum(dim=-1)
        
        # gt_semantic_depths = gt_semantic_depths_cnt[:,1:] / gt_semantic_depths_cnt[:,1:].sum(dim=-1, keepdim=True)
        gt_semantic_depths_sum = gt_semantic_depths_cnt[:,1:].sum(dim=-1, keepdim=True)
        gt_semantic_depths_sum[gt_semantic_depths_sum==0] = 1
        gt_semantic_depths = gt_semantic_depths_cnt[:,1:] / gt_semantic_depths_sum
        # soft_depth_mask = soft_depth_mask & (gt_semantic_depths_sum>0)
        gt_depths[soft_depth_mask] = gt_semantic_depths
  
        return gt_depths, gt_semantics

    @force_fp32()
    def get_depth_and_semantic_loss(self, depth_labels, depth_preds, semantic_labels, semantic_preds=None):
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        semantic_weight = torch.zeros_like(semantic_labels[:,1:2])
        semantic_weight = torch.fill_(semantic_weight, 0.1)
        semantic_weight[semantic_labels[:,1] > 0] = 0.9

        depth_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[depth_mask]
        depth_preds = depth_preds[depth_mask]
        semantic_weight = semantic_weight[depth_mask]

        with autocast(enabled=False):
            depth_loss = (F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ) * semantic_weight).sum() / max(0.1, semantic_weight.sum())

        return self.loss_depth_weight * depth_loss

    def voxel_pooling_prepare_v2(self, coor, kept):
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        # record the index of selected points for acceleration purpose
        ranks_depth = torch.range(
            0, num_points - 1, dtype=torch.int, device=coor.device)
        ranks_feat = torch.range(
            0, num_points // D - 1, dtype=torch.int, device=coor.device)
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()
        # convert coordinate into the voxel space
        coor = ((coor - self.grid_lower_bound.to(coor)) /
                self.grid_interval.to(coor))
        coor = coor.long().view(num_points, 3)
        batch_idx = torch.range(0, B - 1).to(coor).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1)
        coor = torch.cat((coor, batch_idx), 1)
        # filter out points that are outside box
        kept = kept.view(num_points)
        kept &= (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])
        if len(kept) == 0:
            return None, None, None, None, None
        coor, ranks_depth, ranks_feat = \
            coor[kept], ranks_depth[kept], ranks_feat[kept]
        # get tensors from the same voxel next to each other
        ranks_bev = coor[:, 3] * (
            self.grid_size[2] * self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]
        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]
        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
        ), interval_lengths.int().contiguous()

    def voxel_pooling_v2(self, coor, depth, feat, kept):
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor, kept)
        if ranks_feat is None:
            print('warning ---> no points within the predefined '
                  'bev receptive field')
            dummy = torch.zeros(size=[
                feat.shape[0], feat.shape[2],
                int(self.grid_size[2]),
                int(self.grid_size[0]),
                int(self.grid_size[1])
            ]).to(feat)
            dummy = torch.cat(dummy.unbind(dim=2), 1)
            dummy += feat.mean() * 0
            return dummy
        feat = feat.permute(0, 1, 3, 4, 2)
        bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                          int(self.grid_size[1]), int(self.grid_size[0]),
                          feat.shape[-1])  # (B, Z, Y, X, C)
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts,
                               interval_lengths)
        # collapse Z
        bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        return bev_feat

    def view_transform(self, img_feat_shape, input, 
                       depth, gt_depth, tran_feat, 
                       kept, gt_kept, paste_idx, bda_paste):
        B, N, C, H, W = img_feat_shape
        coor = self.get_lidar_coor(*input[1:7])
        gt_bev_feat = None
        if paste_idx is None:
            bev_feat = self.voxel_pooling_v2(
                coor, depth.view(B, N, self.D, H, W),
                tran_feat.view(B, N, self.out_channels, H, W), kept)

            if gt_depth != None:
                gt_bev_feat = self.voxel_pooling_v2(
                    coor, gt_depth.view(B, N, self.D, H, W),
                    tran_feat.view(B, N, self.out_channels, H, W), gt_kept)
                
                bev_feat = torch.cat([bev_feat, gt_bev_feat], 0)
            
        else:
            paste_idx0 = [a[0] for a in paste_idx]
            coor0 = bda_paste.view(B, 1, 1, 1, 1, 3,
                          3).matmul(coor[paste_idx0].unsqueeze(-1)).squeeze(-1)
            bev_feat = self.voxel_pooling_v2(
                coor0, depth.view(B, N, self.D, H, W)[paste_idx0],
                tran_feat.view(B, N, self.out_channels, H, W)[paste_idx0], 
                kept.view(B, N, self.D, H, W)[paste_idx0])

            paste_idx1 = [a[1] for a in paste_idx]
            coor1 = bda_paste.view(B, 1, 1, 1, 1, 3,
                          3).matmul(coor[paste_idx1].unsqueeze(-1)).squeeze(-1)
            bev_feat += self.voxel_pooling_v2(
                coor1, depth.view(B, N, self.D, H, W)[paste_idx1],
                tran_feat.view(B, N, self.out_channels, H, W)[paste_idx1], 
                kept.view(B, N, self.D, H, W)[paste_idx1])   

            if  gt_depth != None:
                gt_bev_feat = self.voxel_pooling_v2(
                    coor0, gt_depth.view(B, N, self.D, H, W)[paste_idx0],
                    tran_feat.view(B, N, self.out_channels, H, W)[paste_idx0], 
                    gt_kept.view(B, N, self.D, H, W)[paste_idx0])
                
                gt_bev_feat += self.voxel_pooling_v2(
                    coor1, gt_depth.view(B, N, self.D, H, W)[paste_idx1],
                    tran_feat.view(B, N, self.out_channels, H, W)[paste_idx1], 
                    gt_kept.view(B, N, self.D, H, W)[paste_idx1]) 
                
                bev_feat = torch.cat([bev_feat, gt_bev_feat], 0)

        return bev_feat

    def get_depth_loss(self, depth, gt_depth, gt_semantic):
        # depth, semantic = img_preds
        depth_labels, semantic_labels = \
            self.get_downsampled_gt_depth_and_semantic(gt_depth, gt_semantic)
        loss_depth = \
            self.get_depth_and_semantic_loss(depth_labels, depth, semantic_labels)
        return loss_depth
    
    @force_fp32()
    def get_bev_loss_normal(self, s_bev, t_bev):
        B, C, H, W = s_bev.shape

        t_norm = torch.norm(t_bev, p=2, dim=1, keepdim=True)
        # s_norm = torch.norm(s_bev, p=2, dim=1, keepdim=True)
        epsilon = 1e-7

        t_bev = t_bev / (t_norm + epsilon)
        s_bev = s_bev / (t_norm + epsilon)

        loss_bev = F.mse_loss(s_bev, t_bev, reduction='none')
        loss_bev = loss_bev.sum() / (B*H*W)

        return loss_bev

    @autocast(enabled=False)
    def forward(self, input):
        (x, rots, trans, intrins, post_rots, post_trans, bda,
         mlp_input, semantic, paste_idx, bda_paste, gt_depth, gt_semantic) = input[:13]

        x = x.float()
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(x, mlp_input) # c = 118(depth) + 80(context)
        depth_digit = x[:, :self.D, ...]
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]
        depth = depth_digit.softmax(dim=1)

        semantic = semantic.view(B * N, 1, H, W)
        kept = (depth >= self.depth_threshold) * (semantic >= self.semantic_threshold)
        
        depth_labels = None
        kept_labels = None
        predgt_depth = None
        predgt_kept = None
        if gt_depth != None:
            depth_labels, semantic_labels = \
                    self.get_downsampled_gt_depth_and_semantic(gt_depth, gt_semantic)
            depth_labels = depth_labels.view(B*N, H, W, -1).permute(0, 3, 1, 2).contiguous()
            semantic_labels = semantic_labels.view(B*N, H, W, -1).permute(0, 3, 1, 2).contiguous()
            replace_mask = (depth_labels.sum(1) == 0) & (semantic[:, 0] >= self.semantic_threshold)
            # pred + gt send to teacher
            predgt_depth = depth_labels.permute(0, 2, 3, 1).contiguous()
            predgt_depth[replace_mask] = depth.permute(0, 2, 3, 1).contiguous()[replace_mask]
            predgt_depth = predgt_depth.permute(0, 3, 1, 2).contiguous()
            predgt_kept = (depth_labels >= self.depth_threshold) * ((semantic_labels[:, 1:2] > 0) | replace_mask.unsqueeze(1))
        
        bev_feat_all = self.view_transform(input[0].shape, input, 
                                           depth, predgt_depth, tran_feat, 
                                           kept, predgt_kept, paste_idx, bda_paste)

        return bev_feat_all, depth