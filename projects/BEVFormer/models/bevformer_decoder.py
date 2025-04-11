import copy
import warnings
import math

import torch 
from torch import nn

from mmengine.model import xavier_init, constant_init

from mmcv.cnn.bricks.transformer import build_feedforward_network
from mmcv.cnn.bricks.norm import build_norm_layer
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
    MultiScaleDeformableAttnFunction_fp16

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class DetectionTransformerDecoder(nn.Module):
    def __init__(self,
                 transformerlayers=None,
                 num_layers=None,
                 return_intermediate=False,
                 **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        transformerlayers = [copy.deepcopy(transformerlayers) for _ in range(num_layers)]
        for i in range(num_layers):
            self.layers.append(DetrTransformerDecoderLayer(**transformerlayers[i]))
        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm
        self.return_intermediate = return_intermediate

    def forward(self,
                query,
                reference_points=None,
                reg_branches=None,
                key_padding_mask=None):
        
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):

            reference_points_input = reference_points[..., :2].unsqueeze(
                2)  # BS NUM_QUERY NUM_LEVEL 2
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                assert reference_points.shape[-1] == 3

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[
                    ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points         



class DetrTransformerDecoderLayer(nn.Module):
    def __init__(self,
                 attn_cfgs=None,
                 ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=256,
                    feedforward_channels=1024,
                    num_fcs=2,
                    ffn_drop=0.,
                    act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):
        deprecated_args = dict(
            feedforward_channels='feedforward_channels',
            ffn_dropout='ffn_drop',
            ffn_num_fcs='num_fcs')
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f'The arguments `{ori_name}` in BaseTransformerLayer '
                    f'has been deprecated, now you should set `{new_name}` '
                    f'and other FFN related arguments '
                    f'to a dict named `ffn_cfgs`. ')
                ffn_cfgs[new_name] = kwargs[ori_name]

        super().__init__()
        
        self.batch_first = batch_first
        num_attn = operation_order.count('self_attn') + operation_order.count('cross_attn')

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = nn.ModuleList()

        # build attention
        index = 0
        for operation_name in operation_order:
            if operation_name == 'self_attn':
                attn_cfgs[index]['batch_first'] = self.batch_first
                attn_cfgs[index].pop('type')
                attention = MultiheadAttention(**attn_cfgs[index])
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1
            elif operation_name == 'cross_attn':
                attn_cfgs[index]['batch_first'] = self.batch_first
                attention = CustomMSDeformableAttention(**attn_cfgs[index])
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1
        
        self.embed_dims = self.attentions[0].embed_dims

        # build ffn
        self.ffns = nn.ModuleList()
        num_ffns = operation_order.count('ffn')
        ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            ffn_cfgs[ffn_index]['embed_dims'] = self.embed_dims
            self.ffns.append(build_feedforward_network(ffn_cfgs[ffn_index], dict(type='FFN')))

        # build norm
        self.norms = nn.ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])


        def forward(self,
                    query,
                    key=None,
                    value=None,
                    query_pos=None,
                    key_pos=None,
                    attn_masks=None,
                    query_key_padding_mask=None,
                    key_padding_mask=None):
            
            norm_index = 0
            attn_index = 0
            ffn_index = 0
            identity = query
            if attn_masks is None:
                attn_masks = [None for _ in range(self.num_attn)]
            elif isinstance(attn_masks, torch.Tensor):
                attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            
            for layer in self.operation_order:
                if layer == 'self_attn':
                    temp_key = temp_value = query
                    query = self.attentions[attn_index](
                        query,
                        temp_key,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=query_key_padding_mask,
                        **kwargs)
                    attn_index += 1
                    identity = query

                elif layer == 'norm':
                    query = self.norms[norm_index](query)
                    norm_index += 1
                
                elif layer == 'cross_attn':
                    query = self.attentions[attn_index](
                        query,
                        key,
                        value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=key_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=key_padding_mask,
                        **kwargs)
                    attn_index += 1
                    identity = query
                
                elif layer == 'ffn':
                    query = self.ffns[ffn_index](
                        query, identity if self.pre_norm else None)
                    ffn_index += 1
            
            return query


class CustomMSDeformableAttention(nn.Module):
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 **kwargs):
        super().__init__()
        # parameters
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.im2col_step = im2col_step
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.batch_first = batch_first

        # modules
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()
    

    def init_weights(self):
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        
        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True
    

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                flag='decoder'):
        
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
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0], -1])
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[: ,:, None, :, None, 2:] \
                * 0.5
        
        if torch.cuda.is_available() and value.is_cuda:

            # using fp16 deformable attention is unstable because it performs many sum operations
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)    

        output = self.output_proj(output)
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        
        return self.dropout(output) + identity
