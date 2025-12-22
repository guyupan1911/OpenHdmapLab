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


@MODELS.register_module()
class BEVFormer(Base3DDetector):

    def __init__(self,
                 img_backbone: ConfigType,
                 img_neck: OptConfigType = None,
                 bev_encoder: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 with_box_refine: bool = False,
                 as_two_stage: bool = False,
                 embed_dims: int = 256,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 video_test_mode: bool = False,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 **kwargs) -> None:
        
        super().__init__(
            data_preprocessor=data_preprocessor,
            init_cfg = init_cfg)
        
        self.embed_dims = embed_dims
  
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
        self._init_layers()
    
    def _init_layers(self) -> None:

        self.bev_encoder = MODELS.build(self.bev_encoder)
            
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
        
        bev_encoder_outputs = self.bev_encoder(
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


