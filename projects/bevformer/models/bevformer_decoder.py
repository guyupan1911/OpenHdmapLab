from typing import Optional

import torch
from torch import nn, Tensor
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention

from mmengine.model import BaseModule, ModuleList
from mmdet.models.layers.transformer import DeformableDetrTransformerDecoderLayer
from mmdet.utils import ConfigType, OptConfigType

from mmhdmap.registry import MODELS


@MODELS.register_module()
class BEVFormerDecoder(BaseModule):

    def __init__(self,
                 num_layers: int,
                 layer_cfg: ConfigType,
                 return_intermediate: bool = True,
                 init_cfg: OptConfigType = None) -> None:
        
        super().__init__(init_cfg=init_cfg)
        self.num_layers = num_layers
        self.layer_cfg = layer_cfg
        self.return_intermediate = return_intermediate
        self._init_layers()
    
    def _init_layers(self) -> None:
        self.layers = ModuleList([
            DeformableDetrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
    
    def forward(self,
                query: Tensor,
                query_pos: Tensor = None,
                vale: Tensor = None,
                key_padding_mask: Tensor = None,
                reference_points: Tensor = None,
                spatial_shapes: Tensor = None,
                level_start_index: Tensor = None,
                reg_branches: Optional[nn.Module] = None,
                **kwargs):
        
        output = query
        intermediate = []
        intermediate_reference_points = []

        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(2)
            output = layer(
                output,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                **kwargs)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                assert reference_points.shape[-1] == 3

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[
                    ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)
        
        return output, reference_points


