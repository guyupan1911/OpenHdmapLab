from typing import Optional
import math
import warnings

import torch
from torch import nn, Tensor

from mmengine.config import ConfigDict
from mmengine.model import BaseModule, constant_init, xavier_init
from mmcv.utils import (IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE)

from .multi_scale_deformable_attn import (
    MultiScaleDeformableAttnFunction,
    multi_scale_deformable_attn_pytorch,
)

from mmhdmap.registry import MODELS


@MODELS.register_module()
class SpatialCrossAttention(BaseModule):

    def __init__(self,
                 embed_dims: int = 256,
                 num_cams: int = 6,
                 dropout: float = 0.1,
                 attn_cfg: dict = dict(
                    type='MSDeformableAttention3D',
                    embed_dims=256,
                    num_levels=4
                 ),
                 init_cfg: Optional[ConfigDict] = None):
        super().__init__(init_cfg)
        self.dropout = nn.Dropout(dropout)
        self.fp16_enabled = False
        self.deformable_attention = MODELS.build(attn_cfg)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weight()

    def init_weight(self):
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
    
    def forward(self,
                query: Tensor,
                key: Optional[Tensor] = None,
                value: Optional[Tensor] = None,
                identity: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None,
                spatial_shapes: Optional[Tensor] = None,
                reference_points_cam: Optional[Tensor] = None,
                bev_mask: Optional[Tensor] = None,
                level_start_index: Optional[Tensor] = None,
                **kwargs):

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        
        slots = torch.zeros_like(query)
    
        bs, num_query, _ = query.size()
        num_Z_anchors = reference_points_cam.size(3)

        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img) #(M)
        max_len = max([len(each) for each in indexes])

        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, num_Z_anchors, 2])
        
        for i in range(bs):
            for j, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indexes[j]
                queries_rebatch[i, j, :len(index_query_per_img)] = query[i, index_query_per_img]
                reference_points_rebatch[i, j, :len(index_query_per_img)] = \
                    reference_points_per_img[i, index_query_per_img]
        
        num_cams, l, bs, embed_dims = key.shape
        key = key.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims)
        
        queries_rebatch = queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims)
        reference_points_rebatch = reference_points_rebatch.view(
            bs * self.num_cams, max_len, num_Z_anchors, 2)
        
        output = self.deformable_attention(
            query=queries_rebatch,
            key=key,
            value=value,
            reference_points=reference_points_rebatch,
            spatial_shapes=spatial_shapes,
            level_start_index = level_start_index)
        
        output = output.view(bs, self.num_cams, max_len, self.embed_dims)

        for i in range(bs):
            for j, index_query_per_img in enumerate(indexes):
                slots[i, index_query_per_img] += output[i, j, :len(index_query_per_img)]
        
        # (num_cams, bs, num_queries, num_Z_anchors)
        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + identity



@MODELS.register_module()
class MSDeformableAttention3D(BaseModule):

    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 num_levels: int = 4,
                 num_points: int = 8,
                 im2col_step: int = 64,
                 dropout: float = 0.1,
                 batch_first: bool = True,
                 norm_cfg: Optional[dict] = None,
                 init_cfg: Optional[ConfigDict] = None):
        super().__init__(init_cfg)
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        
        self.init_weights()    
    
    def init_weights(self):
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        ).view(self.num_heads, 1, 1, 2).repeat(
            1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query: Tensor,
                key: Optional[Tensor] = None,
                value: Optional[Tensor] = None,
                identity: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None,
                reference_points: Optional[Tensor] = None,
                spatial_shapes: Optional[Tensor] = None,
                level_start_index: Optional[Tensor] = None,
                **kwargs):

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = (
                sampling_offsets /
                offset_normalizer[None, None, None, :, None, :])
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels,
                num_all_points // num_Z_anchors, num_Z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)

        elif reference_points.shape[-1] == 4:
            # Not used in this 3D variant
            raise NotImplementedError
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        if ((IS_CUDA_AVAILABLE and value.is_cuda)
                or (IS_MLU_AVAILABLE and value.is_mlu)):
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        if not self.batch_first:
            output = output.permute(1, 0, 2)
        
        return output