from typing import Optional, List

import torch
from torch import Tensor

from mmengine.model import BaseModule, ModuleList
from mmengine.config import ConfigDict
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN

from mmhdmap.registry import MODELS

from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import SpatialCrossAttention

@MODELS.register_module()
class BEVFormerEncoderLayer(BaseModule):

    def __init__(self,
                 temporal_attn_cfg: dict,
                 spatial_cross_attn_cfg: dict,
                 ffn_cfg: dict = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.0,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 norm_cfg: dict = dict(type='LN'),
                 init_cfg: Optional[ConfigDict] = None,
                 batch_first: bool = True,
                 **kwargs) -> None:

        super().__init__(init_cfg)

        self.batch_first = batch_first
        self.temporal_attn_cfg = temporal_attn_cfg
        self.spatial_cross_attn_cfg = spatial_cross_attn_cfg

        # Make sure temporal attention works with (bs, num_query, embed_dims)
        self.temporal_attn_cfg.setdefault('batch_first', self.batch_first)
        self.spatial_cross_attn_cfg.setdefault('batch_first', self.batch_first)
        assert self.temporal_attn_cfg['batch_first'] is True, (
            'First dimension of BEVFormerEncoder is `batch`, please set '
            '`batch_first=True` in temporal_attn_cfg.'
        )
        
        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()
    
    def _init_layers(self) -> None:
        self.temporal_attn = TemporalSelfAttention(**self.temporal_attn_cfg)
        self.embed_dims = self.temporal_attn.embed_dims
        self.spatial_cross_attn = SpatialCrossAttention(**self.spatial_cross_attn_cfg)
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)
    
    def forward(self,
                query: Tensor,
                key: Optional[Tensor] = None,
                value: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, 
                bev_pos: Optional[Tensor] = None,
                key_pos: Optional[Tensor] = None,
                attn_masks: Optional[Tensor] = None,
                query_key_padding_mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None,
                ref_2d: Optional[Tensor] = None,
                ref_3d: Optional[Tensor] = None,
                bev_h: int = 200,
                bev_w: int = 200,
                reference_points_cam: Optional[Tensor] = None,
                bev_mask: Optional[Tensor] = None,
                spatial_shapes: Optional[Tensor] = None,
                level_start_index: Optional[Tensor] = None,
                prev_bev: Optional[Tensor] = None,
                **kwargs):

        query = self.temporal_attn(
            query=query,
            key=prev_bev,
            value=prev_bev,
            identity=None,
            query_pos=bev_pos,
            key_padding_mask=query_key_padding_mask,
            reference_points=ref_2d,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs,
        )
        query = self.norms[0](query)
        query = self.spatial_cross_attn(
            query=query,
            key=key,
            value=value,
            identity=None,
            query_pos=query_pos,
            key_padding_mask=key_padding_mask,
            spatial_shapes=spatial_shapes,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask,
            level_start_index=level_start_index,
            **kwargs,
        )
        query = self.norms[1](query)
        query = self.ffn(query, None)
        query = self.norms[2](query)

        return query