from typing import Optional, Union, List, Tuple, Dict, Sequence
import copy

import numpy as np
from torch import Tensor

from mmdet3d.structures.det3d_data_sample import (Det3DDataSample, SampleList,
                                                  OptSampleList, ForwardResults)
from mmdet3d.utils.typing_utils import (InstanceList, OptConfigType, ConfigType,
                                        OptMultiConfig)

from .base import Base3DDetector
from mmhdmap.registry import MODELS


@MODELS.register_module()
class BEVFormer(Base3DDetector):

    def __init__(self,
                 data_preprocessor: OptConfigType = None,
                 img_backbone: OptConfigType = None,
                 img_neck: OptConfigType = None,
                 bev_encoder: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 video_test_mode: bool = False,
                 init_cfg: OptMultiConfig = None,
                 **kwargs) -> None:
        
        super().__init__(
            data_preprocessor=data_preprocessor,
            init_cfg = init_cfg)
        
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # TODO: use grid mask

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

        self.img_backbone = MODELS.build(img_backbone)

        if img_neck is not None:
            self.img_neck = MODELS.build(img_neck)

        self.bev_encoder = MODELS.build(bev_encoder)

        self.bbox_head = MODELS.build(bbox_head)
    
    def extract_img_feat(self, img: Tensor) -> List[Tensor]:
        """
        Args:
            batch_inputs (Tensor): Image tensor, has shape
                (bs, T, num_cams, C_in, H, W). T is temporal queue and
                T[-1] is current frame.

        Returns:
            List[Tensor]: Multi level feature maps, each has shape
                (bs, T, num_cams, C_out, H, W)
        """
        if img is None:
            return None
        elif img.dim() == 6:
            bs, T, num_cams, C, H, W = img.shape
            img = img.view(bs * T * num_cams, C, H, W)
        elif img.dim() == 5:
            bs, num_cams, C, H, W = img.view(bs * num_cams, C, H, W)

        if self.img_backbone is not None:
            x = self.img_backbone(img)
        if self.img_neck is not None:
            img_feats = self.img_neck(x)
        
        mlvl_img_feats = []
        for img_feat in img_feats:
            _, C, H, W = img_feat.shape
            img_feat_reshape = img_feat.view(bs, -1, num_cams, C, H, W)
            mlvl_img_feats.append(img_feat_reshape)
        
        return mlvl_img_feats

    def extract_feat(self, batch_inputs_dict: dict):
        
        imgs = batch_inputs_dict.get('imgs', None)
        mlvl_img_feats = self.extract_img_feat(imgs)
        
        return mlvl_img_feats

    def loss(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList) -> Union[dict, tuple]:
        pass
        
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
        

        mlvl_temporal_img_feats = self.extract_feat(batch_inputs)
        mlvl_img_feats = [feat[:, -1] for feat in mlvl_temporal_img_feats] # current frame
        
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
        pass


