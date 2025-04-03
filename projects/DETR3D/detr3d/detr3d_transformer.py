import copy
import warnings

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import build_feedforward_network, MultiheadAttention
from mmengine.model import constant_init, xavier_init
from mmengine.config import ConfigDict
from mmdet3d.registry import MODELS

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


class Detr3DTransformer(nn.Module):
    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 decoder=None,
                 **kwargs):
        super().__init__()
        self.decoder = Detr3DTransformerDecoder(**decoder)
        self.embed_dims = self.decoder.embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.two_stage_num_proposals = two_stage_num_proposals
        self._init_layers()
    

    def _init_layers(self):
        self.reference_points = nn.Linear(self.embed_dims, 3)
    

    def forward(self, mlvl_feats, query_embed, reg_branches=None, **kwargs):

        assert query_embed is not None
        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)  # [bs,num_q,c]
        query = query.unsqueeze(0).expand(bs, -1, -1)  # [bs,num_q,c]
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        # decoder
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=mlvl_feats,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            **kwargs)

        inter_references_out = inter_references
        return inter_states, init_reference_out, inter_references_out


class Detr3DTransformerDecoder(nn.Module):
    def __init__(self,
                 num_layers=6,
                 transformerlayers=None,
                 return_intermediate=False,
                 **kwargs):
        print(f'transformerlayers: {transformerlayers}')

        super().__init__()
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        transformerlayers = [
            copy.deepcopy(transformerlayers) for _ in range(num_layers)
        ]
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DetrTransformerDecoderLayer(**transformerlayers[i]))
        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm
    

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                **kwargs):
        
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points
            output = layer(
                output,
                *args,
                reference_points=reference_points,
                **kwargs)
            output = output.permute(1, 0, 2)
            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                assert reference_points.shape[-1] == 3

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(
                    reference_points[..., :2])
                new_reference_points[...,
                                     2:3] = tmp[..., 4:5] + inverse_sigmoid(
                                         reference_points[..., 2:3])
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
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=256,
                    feedforward_channels=1024,
                    num_fcs=2,
                    ffn_drop=0.,
                    act_cfg=dict(type='ReLU', inplace=True),
                 ),
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
                    f'to a dict named `ffn_cfgs`. ', DeprecationWarning)
                ffn_cfgs[new_name] = kwargs[ori_name]

        super().__init__()

        
        assert len(operation_order) == 6
        assert set(operation_order) == set(['self_attn', 'norm', 'cross_attn', 'ffn'])

        self.batch_first = batch_first
        
        num_attn = operation_order.count('self_attn') + operation_order.count('cross_attn')
        assert num_attn == len(attn_cfgs)

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        
        # build attentions
        self.attentions = nn.ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name not in ['self_attn', 'cross_attn']:
                continue

            attn_cfgs[index]['batch_first'] = self.batch_first
            if operation_name == 'self_attn':
                attention = MODELS.build(attn_cfgs[index])
            elif operation_name ==  'cross_attn':
                attention = Detr3DCrossAtten(**attn_cfgs[index])
            attention.operation_name = operation_name
            self.attentions.append(attention)
            index += 1
        
        self.embed_dims = self.attentions[0].embed_dims

        # build ffn
        self.ffns = nn.ModuleList()
        num_ffns = operation_order.count('ffn')
        ffn_cfgs = ConfigDict(ffn_cfgs)
        ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
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
                key_padding_mask=None,
                **kwargs):
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
        else:
            assert len(attn_masks) == self.num_attn
        
        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query
            
            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1
            
            elif layer == 'cross_attn':
                query = self.attentions[norm_index](
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


class Detr3DCrossAtten(nn.Module):
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=5,
                 num_cams=6,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 norm_cfg=None,
                 batch_first=False,
                 **kwargs):
        super().__init__()
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range

        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams
        self.attention_weights = nn.Linear(embed_dims, num_cams * num_levels * num_points)

        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True)
        )
        self.batch_first = batch_first
        self._init_weight()
    

    def _init_weight(self):
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0)


    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                reference_points=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (List[Tensor]): Image features from
                different level. Each element has shape
                (B, N, C, H_lvl, W_lvl).
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            reference_points (Tensor): The normalized 3D reference
                points with shape (bs, num_query, 3)
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        query = query.permute(1, 0, 2)

        bs, num_query, _ = query.size()

        attention_weights = self.attention_weights(query).view(
            bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)
        reference_points_3d, output, mask = feature_sampling(
            value, reference_points, self.pc_range, kwargs['img_metas'])
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)
        attention_weights = attention_weights.sigmoid() * mask
        output = output * attention_weights
        output = output.sum(-1).sum(-1).sum(-1)
        output = output.permute(2, 0, 1)
        # (num_query, bs, embed_dims)
        output = self.output_proj(output)
        pos_feat = self.position_encoder(
            inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)
        return self.dropout(output) + inp_residual + pos_feat


def feature_sampling(mlvl_feats,
                     ref_pt,
                     pc_range,
                     img_metas,
                     no_sampling=False):
    """ sample multi-level features by projecting 3D reference points
            to 2D image
        Args:
            mlvl_feats (List[Tensor]): Image features from
                different level. Each element has shape
                (B, N, C, H_lvl, W_lvl).
            ref_pt (Tensor): The normalized 3D reference
                points with shape (bs, num_query, 3)
            pc_range: perception range of the detector
            img_metas (list[dict]): Meta information of multiple inputs
                in a batch, containing `lidar2img`.
            no_sampling (bool): If set 'True', the function will return
                2D projected points and mask only.
        Returns:
            ref_pt_3d (Tensor): A copy of original ref_pt
            sampled_feats (Tensor): sampled features with shape \
                (B C num_q N 1 fpn_lvl)
            mask (Tensor): Determine whether the reference point \
                has projected outsied of images, with shape \
                (B 1 num_q N 1 1)
    """
    lidar2img = [meta['lidar2img'] for meta in img_metas]
    lidar2img = np.asarray(lidar2img)
    lidar2img = ref_pt.new_tensor(lidar2img)
    ref_pt = ref_pt.clone()
    ref_pt_3d = ref_pt.clone()

    B, num_query = ref_pt.size()[:2]
    num_cam = lidar2img.size(1)
    eps = 1e-5

    ref_pt[..., 0:1] = \
        ref_pt[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]  # x
    ref_pt[..., 1:2] = \
        ref_pt[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]  # y
    ref_pt[..., 2:3] = \
        ref_pt[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]  # z

    # (B num_q 3) -> (B num_q 4) -> (B 1 num_q 4) -> (B num_cam num_q 4 1)
    ref_pt = torch.cat((ref_pt, torch.ones_like(ref_pt[..., :1])), -1)
    ref_pt = ref_pt.view(B, 1, num_query, 4)
    ref_pt = ref_pt.repeat(1, num_cam, 1, 1).unsqueeze(-1)
    # (B num_cam 4 4) -> (B num_cam num_q 4 4)
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4)\
                         .repeat(1, 1, num_query, 1, 1)
    # (... 4 4) * (... 4 1) -> (B num_cam num_q 4)
    pt_cam = torch.matmul(lidar2img, ref_pt).squeeze(-1)

    # (B num_cam num_q)
    z = pt_cam[..., 2:3]
    eps = eps * torch.ones_like(z)
    mask = (z > eps)
    pt_cam = pt_cam[..., 0:2] / torch.maximum(z, eps)  # prevent zero-division
    # padded nuscene image: 928*1600
    (h, w) = img_metas[0]['pad_shape']
    pt_cam[..., 0] /= w
    pt_cam[..., 1] /= h
    # else:
    # (h,w,_) = img_metas[0]['ori_shape'][0]          # waymo image
    # pt_cam[..., 0] /= w # cam0~2: 1280*1920
    # pt_cam[..., 1] /= h # cam3~4: 886 *1920 padded to 1280*1920
    # mask[:, 3:5, :] &= (pt_cam[:, 3:5, :, 1:2] < 0.7) # filter pt_cam_y > 886

    mask = (
        mask & (pt_cam[..., 0:1] > 0.0)
        & (pt_cam[..., 0:1] < 1.0)
        & (pt_cam[..., 1:2] > 0.0)
        & (pt_cam[..., 1:2] < 1.0))

    if no_sampling:
        return pt_cam, mask

    # (B num_cam num_q) -> (B 1 num_q num_cam 1 1)
    mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
    mask = torch.nan_to_num(mask)

    pt_cam = (pt_cam - 0.5) * 2  # [0,1] to [-1,1] to do grid_sample
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat = feat.view(B * N, C, H, W)
        pt_cam_lvl = pt_cam.view(B * N, num_query, 1, 2)
        sampled_feat = F.grid_sample(feat, pt_cam_lvl)
        # (B num_cam C num_query 1) -> List of (B C num_q num_cam 1)
        sampled_feat = sampled_feat.view(B, N, C, num_query, 1)
        sampled_feat = sampled_feat.permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)

    sampled_feats = torch.stack(sampled_feats, -1)
    # (B C num_q num_cam fpn_lvl)
    sampled_feats = \
        sampled_feats.view(B, C, num_query, num_cam, 1, len(mlvl_feats))
    return ref_pt_3d, sampled_feats, mask
