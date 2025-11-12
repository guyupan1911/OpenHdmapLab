from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from mmcv import ops
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.utils import ConfigType, OptMultiConfig


class BaseRoIExtractor(BaseModule, metaclass=ABCMeta):
    
    def __init__(self,
                 roi_layer: ConfigType,
                 out_channels: int,
                 featmap_strides: List[int],
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
    
    @property
    def num_inputs(self) -> int:
        return len(self.featmap_strides)

    def build_roi_layers(self, layer_cfg: ConfigType,
                         featmap_strides: List[int]) -> nn.ModuleList:
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        if isinstance(layer_type, str):
            assert hasattr(ops, layer_type)
            layer_cls = getattr(ops, layer_type)
        else:
            layer_cls = layer_type
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers
    
    def roi_rescale(self, rois: Tensor, scale_factor: float) -> Tensor:
        
        cx = (rois[:, 1] + rois[:, 3]) * 0.5
        cy = (rois[:, 2] + rois[:, 4]) * 0.5
        w = rois[:, 3] - rois[:, 1]
        h = rois[:, 4] - rois[:, 2]
        new_w = w * scale_factor
        new_h = h * scale_factor
        x1 = cx - new_w * 0.5
        x2 = cx + new_w * 0.5
        y1 = cy - new_h * 0.5
        y2 = cy + new_h * 0.5
        new_rois = torch.stack((rois[:, 0], x1, y1, x2, y2), dim=-1)
        return new_rois

    @abstractmethod
    def forward(self,
                feats: Tuple[Tensor],
                rois: Tensor,
                roi_scale_factor: Optional[float] = None) -> Tensor:
        pass