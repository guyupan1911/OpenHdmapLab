import copy

import torch
from torch import nn

from mmdet3d.registry import MODELS

from .bevformer_head import BEVFormerHead

class BEVFormer(nn.Module):
    def __init__(self,
                 use_grid_mask=True,
                 video_test_mode=True,
                 pretrained=dict(img='torchvision://resnet50'),
                 img_backbone=None,
                 img_neck=None,
                 pts_bbox_head=None,
                 train_cfg=None,
                 **kwargs):
        super().__init__()

        # temporal parameters
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0.
        }


        if img_backbone:
            self.img_backbone = MODELS.build(img_backbone)
        if img_neck:
            self.img_neck = MODELS.build(img_neck)
        if pts_bbox_head:
            self.pts_bbox_head = BEVFormerHead(**pts_bbox_head)

    def extract_feat(self, img, img_metas=None, len_queue=None):
        """
            Args:
                img: Tensor(bs, queue_length, num_cams, C, H, W)
        """
        B = img.size(0)
        Q = img.size(1)
        if img is not None:
            B, Q, N, C, H, W = img.size()
            img = img.reshape(B*Q*N, C, H, W)
            # if self.use_grid_mask:
            #     img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        
        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BQN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, Q, -1, C, H, W))
        
        return img_feats_reshaped
        

    def forward(self, img, img_metas, return_loss=True):
        if return_loss:
            pass
        else:
            return self.forward_test(img, img_metas)


    def forward_test(self, img, img_metas):
        """
            Args:
                img: Tensor(bs, quene_length, num_cams, C, H, W)
                img_metas: dict of img_metas 
        """
        if img_metas[0]['scene_token'][0] != self.prev_frame_info['scene_token']:
            self.prev_frame_info['prev_bev'] = None
        self.prev_frame_info['scene_token'] = img_metas[0]['scene_token'][0]

        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        tmp_pos = copy.deepcopy(img_metas[0]['can_bus'][0][:3])
        tmp_angle = copy.deepcopy(img_metas[0]['can_bus'][0][-1])
        # print(f'tmp_pos: {tmp_pos}, tmp_angle: {tmp_angle}')
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0]['can_bus'][0][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0]['can_bus'][0][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0]['can_bus'][0][:3] = 0
            img_metas[0]['can_bus'][0][-1] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img, img_metas, prev_bev=self.prev_frame_info['prev_bev'])

        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['tmp_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev


    def simple_test_pts(self, img_feats, img_metas, prev_bev=None, rescale=False):
        """
            Args:
                img_feats: List of Tensor, multi level img feats
        """
        print(f'img_feats: {img_feats[0].shape}')
        outs = self.pts_bbox_head(img_feats, img_metas, prev_bev=prev_bev)


    def simple_test(self, img, img_metas, prev_bev=None, rescale=False):
        """
            Args:
                img Tensor(bs, queue_length, num_cams, C, H, W)
                img_metas: dict with keys (0, queue_length)
        """

        # extract feats
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        # pts_head
        print(f'img_metas: {len(img_metas)}')
        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale)

        # for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
        #     result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list