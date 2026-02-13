from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmengine.model import BaseModule
from mmengine.model import caffe2_xavier_init
from mmengine.structures import InstanceData, PixelData

from mmdet.structures import SampleList
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptMultiConfig, reduce_mean)

from mmhdmap.registry import MODELS, TASK_UTILS



@MODELS.register_module()
class Mask2FormerHead(BaseModule):

    def __init__(self,
                 feat_channels: int,
                 out_channels: int,
                 num_things_classes: int = 80,
                 num_stuff_classes: int = 53,
                 num_queries: int = 100,
                 loss_cls: ConfigType = dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0,
                    class_weight=[1.0] * 133 + [0.1]),
                 loss_mask: ConfigType = dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=20.0),
                 loss_dice: ConfigType = dict(
                    type='DiceLoss',
                    use_sigmoid=True,
                    activate=True,
                    naive_dice=True,
                    loss_weight=1.0),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.num_queries = num_queries
        self.feat_channels = feat_channels
        self.out_channels = out_channels

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))
        
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = TASK_UTILS.build(train_cfg['assigner'])
            self.sampler = TASK_UTILS.build(
                train_cfg['sampler'], default_args=dict(context=self))
        
        self.class_weight = loss_cls.class_weight
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_mask = MODELS.build(loss_mask)
        self.loss_dice = MODELS.build(loss_dice)


    def forward(self,
                cls_pred_list: Optional[List[Tensor]],
                mask_pred_list: Optional[List[Tensor]],
                **kwargs) -> Tuple[List[Tensor], List[Tensor]]:
        
        return cls_pred_list, mask_pred_list
    
    def predict(self,
                cls_pred_list: List[Tensor],
                mask_pred_list: List[Tensor],
                batch_data_samples: SampleList,
                **kwargs) -> Tuple[Tensor]:
        
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        mask_cls_results = cls_pred_list[-1]
        mask_pred_results = mask_pred_list[-1]

        # upsample masks
        img_shape = batch_img_metas[0]['batch_input_shape']
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(img_shape[0], img_shape[1]),
            mode='bilinear',
            align_corner=False)
        
        return mask_cls_results, mask_pred_results



        