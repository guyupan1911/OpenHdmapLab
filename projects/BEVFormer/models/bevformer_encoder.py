import copy
import warnings

import torch
from torch import nn

from mmcv.cnn.bricks.transformer import build_feedforward_network
from mmcv.cnn.bricks.norm import build_norm_layer
from .spatial_cross_attention import SpatialCrossAttention
from .temporal_self_attention import TemporalSelfAttention


class BEVFormerEncoder(nn.Module):
    def __init__(self,
                 num_layers=None,
                 pc_range=None,
                 num_points_in_pillar=4,
                 return_intermediate=False,
                 transformerlayers=None,
                 dataset_type='nuscenes',
                 **kwargs):
        super().__init__()

        # parameters
        self.num_layers = num_layers
        self.pc_range = pc_range
        self.num_points_in_pillar = num_points_in_pillar

        # build transformer layers
        transformerlayers = [
            copy.deepcopy(transformerlayers) for _ in range(num_layers)
        ]

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(BEVFormerLayer(**transformerlayers[i]))
        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm


class BEVFormerLayer(nn.Module):
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
                 batch_first=True,
                 **kwargs):
        super().__init__()
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
        
        self.batch_first = batch_first

        num_attn = operation_order.count('self_attn') + operation_order.count('cross_attn')
        # attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = nn.ModuleList()


        index = 0
        for operation_name in operation_order:
            if operation_name == 'self_attn':
                attn_cfgs[index]['batch_first'] = self.batch_first
                attention = TemporalSelfAttention(**attn_cfgs[index])
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1
            elif operation_name == 'cross_attn':
                attn_cfgs[index]['batch_first'] = self.batch_first
                attention = SpatialCrossAttention(**attn_cfgs[index])
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = nn.ModuleList()
        num_ffns = operation_order.count('ffn')
        ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns

        for ffn_index in range(num_ffns):
            ffn_cfgs[ffn_index]['embed_dims'] = self.embed_dims

            self.ffns.append(
               build_feedforward_network(ffn_cfgs[ffn_index]))

        self.norms = nn.ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])