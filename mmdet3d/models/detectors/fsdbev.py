# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32, load_checkpoint

# from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from .. import builder
from .centerpoint import CenterPoint
from mmdet.models.backbones.resnet import ResNet
from torch.cuda.amp import autocast
from ...datasets.pipelines.loading import LoadAnnotationsBEVDepth

from .bevdet import BEVDet


@DETECTORS.register_module()
class FSDBEV(BEVDet):
    
    def __init__(self, 
                 use_bev_paste,
                 use_bev_distll,
                 bda_aug_conf,
                 pre_process=None,
                 align_after_view_transfromation=False,
                 num_adj=1,
                 with_prev=True,
                 **kwargs):
        super(FSDBEV, self).__init__(**kwargs)
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)
        self.align_after_view_transfromation = align_after_view_transfromation
        self.num_frame = num_adj + 1
        self.with_prev = with_prev
        self.grid = None 
        self.use_bev_distll = use_bev_distll
        self.use_bev_paste = use_bev_paste
        if use_bev_paste:
            self.loader = LoadAnnotationsBEVDepth(bda_aug_conf, None, is_train=True)

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        if self.with_img_neck:
            x = self.img_neck(x)

        x = [t.view(B, N, t.shape[-3], t.shape[-2], t.shape[-1]) for t in x]

        if self.with_img_roi_head:
            preds_hms = self.img_roi_head(x)
            x, hms_s = self.img_roi_head.get_aug_feature(preds_hms, x)

        return x, preds_hms, hms_s

    def extract_feat(self, points, img, img_metas, **kwargs):
        img_feats, depth, preds_hms = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None
        return (img_feats, pts_feats, depth, preds_hms)

    def prepare_inputs(self, inputs, stereo=False):
        # split the inputs into each frame
        B, N, C, H, W = inputs[0].shape
        N = N // self.num_frame
        imgs = inputs[0].view(B, N, self.num_frame, C, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs[1:7]

        sensor2egos = sensor2egos.view(B, self.num_frame, N, 4, 4)
        ego2globals = ego2globals.view(B, self.num_frame, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0, 1, ...].unsqueeze(1).unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()


        extra = [
            sensor2keyegos,
            ego2globals,
            intrins.view(B, self.num_frame, N, 3, 3),
            post_rots.view(B, self.num_frame, N, 3, 3),
            post_trans.view(B, self.num_frame, N, 3)
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        sensor2keyegos, ego2globals, intrins, post_rots, post_trans = extra
        
        if len(inputs) >= 9:
            paste_idx = inputs[7]
            bda_paste = inputs[8]
        else:
            paste_idx = None
            bda_paste = None

        if len(inputs) == 11:
            gt_depths_adj = inputs[9][1].view(B, N, self.num_frame-1, H, W)
            gt_depths_adj = torch.split(gt_depths_adj, 1, 2)
            gt_depths_adj = [t.squeeze(2) for t in gt_depths_adj]
            gt_depths = [inputs[9][0]] + gt_depths_adj
            
            gt_segs_adj = inputs[10][1].view(B, N, self.num_frame-1, H, W)
            gt_segs_adj = torch.split(gt_segs_adj, 1, 2)
            gt_segs_adj = [t.squeeze(2) for t in gt_segs_adj]
            gt_segs = [inputs[10][0]] + gt_segs_adj
        else:
            gt_depths = [None] * self.num_frame
            gt_segs = [None] * self.num_frame

        return imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
               bda, paste_idx, bda_paste, gt_depths, gt_segs

    def prepare_bev_feat(self, img, rot, tran, intrin, post_rot, post_tran,
                         bda, mlp_input, paste_idx, bda_paste, gt_depth, gt_seg):
        x, preds_hms, hms_s = self.image_encoder(img)
        bev_feat, depth = self.img_view_transformer(
            [x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input, hms_s, paste_idx, bda_paste, gt_depth, gt_seg])
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth, preds_hms

    def extract_img_feat(self,
                         img,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, paste_idx, bda_paste, gt_depths, gt_segs = self.prepare_inputs(img)
        """Extract features of images."""
        bev_feat_list = []
        key_frame = True  # back propagation for key frame only
        for img, sensor2keyego, ego2global, intrin, post_rot, post_tran, gt_depth, gt_seg in zip(
                imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, gt_depths, gt_segs):
            if True:
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin, post_rot, post_tran, bda)
                inputs_curr = (img, sensor2keyego, ego2global, intrin, post_rot,
                               post_tran, bda, mlp_input, paste_idx, bda_paste, gt_depth, gt_seg)
                if key_frame:
                    bev_feat, depth, preds_hms = self.prepare_bev_feat(*inputs_curr)
                else:
                    with torch.no_grad():
                        bev_feat, _, _ = self.prepare_bev_feat(*inputs_curr)
            else:
                bev_feat = torch.zeros_like(bev_feat_list[0])
            bev_feat_list.append(bev_feat)
            key_frame = False
        
        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth, preds_hms

    @force_fp32()
    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]

        return x

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        img_feats, _, _, _ = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        assert self.with_pts_bbox
        outs = self.pts_bbox_head(img_feats)
        return outs

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        img_feats, _, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        if self.use_bev_paste:
            B = len(gt_bboxes_3d)
            paste_idx = []
            for i in range(B):
                for j in range(i, i + 1):
                    if j+1>=B: j-=B
                    paste_idx.append([i, j+1])
            
            gt_boxes_paste = []
            gt_labels_paste = []
            bda_mat_paste = []
            for i in range(len(paste_idx)):
                gt_boxes_tmp = []
                gt_labels_tmp = []
                for j in paste_idx[i]:
                    gt_boxes_tmp.append(gt_bboxes_3d[j])
                    gt_labels_tmp.append(gt_labels_3d[j])
                gt_boxes_tmp = torch.cat([tmp.tensor for tmp in gt_boxes_tmp], dim=0)
                gt_labels_tmp = torch.cat(gt_labels_tmp, dim=0)
                rotate_bda, scale_bda, flip_dx, flip_dy = self.loader.sample_bda_augmentation()
                gt_boxes_tmp, bda_rot = self.loader.bev_transform(gt_boxes_tmp.cpu(), rotate_bda, scale_bda, flip_dx, flip_dy)
                gt_boxes_tmp = gt_bboxes_3d[0].new_box(gt_boxes_tmp.cuda())
                bda_mat_paste.append(bda_rot.cuda())
                gt_boxes_paste.append(gt_boxes_tmp)
                gt_labels_paste.append(gt_labels_tmp)
            gt_bboxes_3d = gt_boxes_paste
            gt_labels_3d = gt_labels_paste
            img_inputs.append(paste_idx)
            img_inputs.append(torch.stack(bda_mat_paste))
        else:
            img_inputs.append(None)
            img_inputs.append(None)
        
        if self.use_bev_distll:
            img_inputs.append([kwargs['gt_depth'].clone(), kwargs['gt_depth_adj'].clone()])
            img_inputs.append([kwargs['gt_semantic'].clone(), kwargs['gt_semantic_adj'].clone()])
        else:
            img_inputs.append([None, None])
            img_inputs.append([None, None])

        img_feats, pts_feats, depth, preds_hms = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)

        gt_depth = kwargs['gt_depth']
        gt_semantic = kwargs['gt_semantic']
        loss_depth = self.img_view_transformer.get_depth_loss(depth, gt_depth, gt_semantic)
        loss_bev = \
            self.img_view_transformer.get_bev_loss_normal(img_feats[0][:gt_depth.shape[0]], 
                                                        img_feats[0][gt_depth.shape[0]:])
        losses = dict(loss_depth=loss_depth, loss_bev=loss_bev)
        
        loss_ct = self.img_roi_head.get_loss(preds_hms, kwargs['hms'])
        losses.update(loss_ct)

        with autocast(False):
            gt_bboxes_3d.extend(gt_bboxes_3d)
            gt_labels_3d.extend(gt_labels_3d)
            img_metas.extend(img_metas)
            losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
        losses.update(losses_pts)

        return losses
    
    