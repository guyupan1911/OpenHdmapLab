import math
import warnings
from typing import Optional, no_type_check
import torch
from torch import nn
import torch.nn.functional as F

from mmengine.config import ConfigDict
from mmengine.model import BaseModule, constant_init, xavier_init
from mmhdmap.registry import MODELS
from mmengine.utils import deprecated_api_warning

from mmcv.utils import (IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE)

from .multi_scale_deformable_attn import (MultiScaleDeformableAttnFunction,
    multi_scale_deformable_attn_pytorch)


@MODELS.register_module()
class TemporalSelfAttention(BaseModule):

    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 num_levels: int = 4,
                 num_points: int = 4,
                 num_bev_queue: int = 2,
                 im2col_step: int = 64,
                 dropout: float = 0.1,
                 batch_first: bool = True,
                 norm_cfg: Optional[dict] = None,
                 init_cfg: Optional[ConfigDict] = None,
                 value_proj_ratio: float = 1.0,
                 **kwargs):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, ' 
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

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
        self.num_bev_queue = num_bev_queue
        self.sampling_offsets = nn.Linear(
            embed_dims * self.num_bev_queue,
            self.num_bev_queue * num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(
            embed_dims * self.num_bev_queue,
            self.num_bev_queue * num_heads * num_levels * num_points)
        value_proj_size = int(embed_dims * value_proj_ratio)
        self.value_proj = nn.Linear(embed_dims, value_proj_size)
        self.output_proj = nn.Linear(value_proj_size, embed_dims)
        self.init_weights()
    
    def init_weights(self) -> None:
        constant_init(self.sampling_offsets, 0.)
        device = next(self.parameters()).device
        # (num_heads,)
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32,
            device=device) * (2.0 * math.pi / self.num_heads)
        # (num_heads, 2)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # (num_heads, 1, 1, 2)
        # -> (num_heads, num_queue * num_levels, num_points, 2)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                        self.num_heads, 1, 1, 2
                     ).repeat(1, self.num_bev_queue * self.num_levels, self.num_points, 1)
        # grid_init (num_heads, num_queue * num_levels, num_points, 2)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True
    
    @no_type_check
    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='TemporalSelfAttention')
    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                identity: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                reference_points: Optional[torch.Tensor] = None, # (bs, sum(hw), num_levels, 2)
                spatial_shapes: Optional[torch.Tensor] = None,
                level_start_index: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
                
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            if value is not None:
                value = value.permute(1, 0, 2)

        if value is None:
            # without history bev feature, use query twice
            bs, num_queries, embed_dims = query.shape
            value = torch.stack([query, query], 1).reshape(bs * 2, num_queries, embed_dims)
        
        bs, num_query, _ = query.shape
        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        assert self.num_bev_queue == 2

        # (bs, num_queries, embed_dims * 2)
        query = torch.cat([value[:bs], query], -1)

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs * self.num_bev_queue, num_value, self.num_heads, -1)
        
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points)
        
        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)


        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1) # (num_levels, 2)
            sampling_locations = (reference_points[:, :, None, :, None, :]
                + sampling_offsets 
                / offset_normalizer[None, None, None, :, None, :])
            # (bs, num_queries, heads, levels, points, 2)
        elif reference_points.shape[-1] == 4:
            sampling_locations = (reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5)
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f'2 or 4, but get {reference_points.shape[-1]} instead.')
        if ((IS_CUDA_AVAILABLE and value.is_cuda)
                or (IS_MLU_AVAILABLE and value.is_mlu)):
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        
        output = output.permute(1, 2, 0)
        output = output.view(num_query, self.embed_dims, bs, self.num_bev_queue)
        output = output.mean(-1)

        output = output.permute(2, 0, 1)

        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)
        
        return self.dropout(output) + identity