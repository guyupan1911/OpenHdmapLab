from typing import Optional, Union, List, Tuple, Dict, Sequence
import copy

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.structures.det3d_data_sample import (Det3DDataSample, SampleList,
                                                  OptSampleList, ForwardResults)
from mmdet3d.utils.typing_utils import InstanceList
from mmdet.utils import OptConfigType, ConfigType, OptMultiConfig
from torchvision.transforms.functional import rotate

from mmhdmap.registry import MODELS


class BEVFormer(Base3DDetector):

    def __init__(self,
                 img_backbone: ConfigType,
                 img_neck: OptConfigType = None,
                 bev_encoder: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 positional_encoding: OptConfigType = None,
                 with_box_refine: bool = False,
                 as_two_stage: bool = False,
                 embed_dims: int = 256,
                 num_feature_levels: int = 4,
                 bev_h: int = 30,
                 bev_w: int = 30,
                 num_cams: int = 6,
                 use_cams_embeds: bool = True,
                 rotate_center: Optional[Sequence[float]] = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 video_test_mode: bool = False,
                 use_can_bus=True,
                 can_bus_norm=True,
                 pc_range=None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 **kwargs) -> None:
        
        super().__init__(
            data_preprocessor=data_preprocessor,
            init_cfg = init_cfg)
        
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.use_cams_embeds = use_cams_embeds
        # torchvision.transforms.functional.rotate expects center=(x, y) in pixel coords.
        # If not provided, use the BEV feature map center for stability across bev_h/bev_w.
        if rotate_center is None:
            self.rotate_center = ((self.bev_w - 1) / 2.0, (self.bev_h - 1) / 2.0)
        else:
            assert len(rotate_center) == 2, 'rotate_center must be a 2-tuple/list: (x, y)'
            self.rotate_center = (float(rotate_center[0]), float(rotate_center[1]))
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.pc_range = pc_range

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.bev_encoder = bev_encoder
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage

        # TODO: use grid mask

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

        # init layers
        self.img_backbone = MODELS.build(img_backbone)
        if img_neck is not None:
            self.img_neck = MODELS.build(img_neck)
        self.bbox_head = MODELS.build(bbox_head)
        self.positional_encoding = MODELS.build(positional_encoding)
        self._init_layers()
    
    def _init_layers(self) -> None:

        self.bev_encoder = MODELS.build(self.bev_encoder)

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            
        
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))

        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True)
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def extract_img_feat(self, batch_inputs: Tensor) -> List[Tensor]:
        """
        Args:
            batch_inputs (Tensor): Image tensor, has shape
                (bs, T, num_cams, C_in, H, W). T is temporal queue and
                T[-1] is current frame.

        Returns:
            List[Tensor]: Multi level feature maps, each has shape
                (bs, T, num_cams, C_out, H, W)
        """
        assert batch_inputs.dim() == 6
        bs, T, num_cams, dim, H, W = batch_inputs.shape
        batch_inputs = batch_inputs.view(bs * T * num_cams, dim, H, W)

        x = self.img_backbone(batch_inputs)
        if self.img_neck is not None:
            img_feats = self.img_neck(x)
        
        multi_level_img_feats = []
        for img_feat in img_feats:
            _, C, H, W = img_feat.shape
            img_feat_reshape = img_feat.view(bs, T, num_cams, C, H, W)
            multi_level_img_feats.append(img_feat_reshape)
        
        return multi_level_img_feats

    def extract_feat(self, batch_inputs_dict: dict):
        assert 'imgs' in batch_inputs_dict
        return self.extract_img_feat(batch_inputs_dict['imgs'])

    def loss(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList) -> Union[dict, tuple]:
        """
        Args:

        Returns:
            dict: A dictionary of loss component
        """


        img_feats = self.extract_img_feat(batch_inputs)
        head_inputs_dict = self.forward_bev_encoder(img_feats,
                                                    batch_data_samples)
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        
        return losses
        
    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        """
        Args:
            batch_inputs (Tensor): Image tensor, has shape
                (bs, T, num_cams, C_in, H, W). T is temporal queue and
                T[-1] is current frame.
        
        Results:
            List[Det3dDataSample]: Detection results of input
        """
        
        # temporal: keep prev_bev and ego motion across frames
        if batch_data_samples[0].metainfo['scene_token'] != self.prev_frame_info['scene_token']:
            self.prev_frame_info['prev_bev'] = None
        
        self.prev_frame_info['scene_token'] = batch_data_samples[0].metainfo['scene_token']

        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None
        
        # x, y, yaw
        tmp_pos = copy.deepcopy(batch_data_samples[0].metainfo['can_bus'][:3])
        tmp_angle = copy.deepcopy(batch_data_samples[0].metainfo['can_bus'][-1])

        if self.prev_frame_info['prev_bev'] is not None:
            batch_data_samples[0].metainfo['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            batch_data_samples[0].metainfo['can_bus'][-1] -= self.prev_frame_info['prev_angle']
            # Normalize delta yaw to avoid 0/360 wrap-around (e.g., 359 -> 1 deg).
            # Keep it in [-180, 180] degrees, consistent with small inter-frame rotations.
            delta_yaw = float(batch_data_samples[0].metainfo['can_bus'][-1])
            delta_yaw = (delta_yaw + 180.0) % 360.0 - 180.0
            batch_data_samples[0].metainfo['can_bus'][-1] = delta_yaw
        else:
            batch_data_samples[0].metainfo['can_bus'][0:3] = 0
            batch_data_samples[0].metainfo['can_bus'][-1] = 0
        

        # only use current frame
        mlvl_img_feats = self.extract_feat(batch_inputs)
        
        bev_encoder_outputs = self.forward_bev_encoder(
            mlvl_img_feats, batch_data_samples, prev_bev=self.prev_frame_info['prev_bev'])
       
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = bev_encoder_outputs['bev_embed']

        bev_embed = bev_encoder_outputs['bev_embed']

        results_list_3d = self.bbox_head.predict(bev_embed, batch_data_samples=batch_data_samples)

        detsamples = self.add_pred_to_datasample(
            batch_data_samples, data_instances_3d=results_list_3d)
        
        return detsamples

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        img_feats = self.extract_img_feat(batch_inputs)
        head_inputs_dict = self.forward_bev_encoder(img_feats,
                                                    batch_data_samples)
        results = self.bbox_head.forward(**head_inputs_dict)
        return results

    def forward_bev_encoder(self, mlvl_feats, batch_data_samples, prev_bev) -> Tensor:
        """
        Args:

        Returns:
            bev_embedding: bev features after temporal self attn and spatial cross attn,
                has shape (bs, bev_h*bev_w, embed_dims)
        """

        mlvl_feats = [feat[:, -1] for feat in mlvl_feats] # current frame

        bs, num_cams, _, _, _ = mlvl_feats[0].shape
    
        bev_query = self.bev_embedding.weight # (bev_h*bev_w, embed_dims)
        bev_mask = bev_query.new_zeros((bs, self.bev_h, self.bev_w)) 
        bev_pos = self.positional_encoding(bev_mask) # (1, embed_dims, bev_h, bev_w)

        # -> (bev_h*bev_w, bs, embed_dims)
        bev_query = bev_query.unsqueeze(1).repeat(1, bs, 1)
        # -> (bev_h*bev_w, 1, embed_dims)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # debug prints removed


        # obtain rotation angle and shift with ego motion
        delta_x = np.array([each.metainfo['can_bus'][0] for each in batch_data_samples])
        delta_y = np.array([each.metainfo['can_bus'][1] for each in batch_data_samples])
        # BEVFormer can_bus convention:
        # - can_bus[-2]: absolute ego yaw in radians (used for shift direction)
        # - can_bus[-1]: delta ego yaw in degrees (used for rotating prev_bev)
        ego_angle = np.array(
            [each.metainfo['can_bus'][-2] / np.pi * 180 for each in batch_data_samples])

        # pc_range = [xmin, ymin, zmin, xmax, ymax, zmax]
        pc_range = getattr(self.bev_encoder, 'pc_range', None) or getattr(self, 'pc_range', None)
        assert pc_range is not None, 'pc_range must be provided by bev_encoder or model config'
        grid_length_y = (pc_range[4] - pc_range[1]) / self.bev_h
        grid_length_x = (pc_range[3] - pc_range[0]) / self.bev_w
        translation_length = np.sqrt(delta_x**2 + delta_y**2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / self.bev_h
        shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / self.bev_w

        shift = bev_query.new_tensor(
            [shift_x, shift_y]).permute(1, 0)
        
        if prev_bev is not None:
            if prev_bev.shape[1] == self.bev_h * self.bev_w:
                # -> (bev_h*bev_w, bs, embed_dims)
                prev_bev = prev_bev.permute(1, 0, 2)
            for i in range(bs):
                rotation_angle = batch_data_samples[i].metainfo['can_bus'][-1]
                tmp_prev_bev = prev_bev[:, i].reshape(
                    self.bev_h, self.bev_w, -1).permute(2, 0, 1)
                # Guard against a misconfigured rotate_center (must lie within the BEV map).
                if not (0.0 <= self.rotate_center[0] <= (self.bev_w - 1) and
                        0.0 <= self.rotate_center[1] <= (self.bev_h - 1)):
                    raise ValueError(
                        f'rotate_center={self.rotate_center} is outside the BEV map '
                        f'(bev_w={self.bev_w}, bev_h={self.bev_h}). '
                        'Set rotate_center=None to use the BEV center, or pass a valid (x, y).'
                    )
                tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                      center=self.rotate_center)
                tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                    self.bev_h * self.bev_w, 1, -1)
                prev_bev[:, i] = tmp_prev_bev[:, 0]

        can_bus = bev_query.new_tensor(
            [each.metainfo['can_bus'] for each in batch_data_samples])
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]

        bev_query = bev_query + self.use_can_bus * can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cams, C, H, W = feat.shape
            spatial_shape = (H, W)
            # -> (num_cams, bs, H*W, C)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :]
            feat = feat + self.level_embeds[None, None, lvl:lvl+1, :]
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
        
        # -> (num_cams, sum(HW), bs, embed_dims)
        feat_flatten = torch.cat(feat_flatten, 2)
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)

        # -> (num_levels, 2）
        spatial_shapes = bev_pos.new_tensor(spatial_shapes, dtype=torch.long)
        level_sizes = spatial_shapes.prod(dim=1)
        level_start_index = torch.cat([level_sizes.new_zeros(1), level_sizes.cumsum(0)[:-1]])

        bev_embed = self.bev_encoder(
            bev_query=bev_query,
            key=feat_flatten,
            value=feat_flatten,
            bev_pos=bev_pos,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=None,
            shift=shift,
            batch_data_samples=batch_data_samples)
        

        return {
            "bev_embed": bev_embed
        }
