import copy 
from typing import Optional, List, Tuple

import torch
from torch import Tensor
import numpy as np

from mmengine.model import BaseModule, ModuleList
from mmengine.config import ConfigDict
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN

from mmhdmap.registry import MODELS

from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import SpatialCrossAttention


@MODELS.register_module()
class BEVFormerEncoder(BaseModule):

    def __init__(self, 
                 pc_range: Tuple[int],
                 num_points_in_pillar: int = 4,
                 return_intermediate: bool = False,
                 num_layers: int = 6,
                 num_cp: int = -1,
                 layer_cfg: Optional[ConfigDict] = None,
                 init_cfg: Optional[ConfigDict] = None) -> None:

        super().__init__(init_cfg=init_cfg)
        self.fp16_enabled = False
        self.pc_range = pc_range
        self.num_points_in_pillar = num_points_in_pillar
        self.return_intermediate = return_intermediate
        self.num_layers = num_layers
        self.layer_cfg = layer_cfg
        self.num_cp = num_cp
        assert self.num_cp <= self.num_layers
        self._init_layers()
    
    def _init_layers(self) -> None:
        self.layers = ModuleList([
            BEVFormerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])

        self.embed_dims = self.layers[0].embed_dims
    
    @staticmethod
    def get_reference_points(
        H: int,
        W: int,
        Z: int = 8,
        num_points_in_pillar: int = 4,
        dim: str = '3d', # ['3d', '2d']
        bs: int = 1,
        device: str = 'cuda',
        dtype: torch.dtype = torch.float) -> Tensor:

        if dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack([ref_x, ref_y], -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d # (bs, sum(HW), 1, 2)

        elif dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack([xs, ys, zs], -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d # (bs, num_points_in_pillar, sum(HW), 3)

    # @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, pc_range, batch_data_samples):
        """
        Args:
            reference_points: 3d points on bev plane, has shape
                (bs, num_points_in_pillar, sum(HW), 3)
            pc_range: lidar range
            batch_data_samples: metainfo
        """

        lidar2img = []
        for data_sample in batch_data_samples:
            lidar2img.append(copy.deepcopy(data_sample.metainfo['lidar2img']))
        lidar2img = np.asarray(lidar2img)
        print(f'lidar2img: {lidar2img.shape}')
        lidar2img = reference_points.new_tensor(lidar2img) # (bs, num_cams, 4, 4)
        reference_points = reference_points.clone()

        # convert to real-world coordinate
        reference_points[..., 0:1] = (reference_points[..., 0:1] *
                                      (pc_range[3] - pc_range[0]) + pc_range[0])
        reference_points[..., 1:2] = (reference_points[..., 1:2] * 
                                      (pc_range[4] - pc_range[1]) + pc_range[1])
        reference_points[..., 2:3] = (reference_points[..., 2:3] *
                                      (pc_range[5] - pc_range[2]) + pc_range[2])

        # -> (bs, num_points_in_pillar, sum(hw), 4)
        reference_points = torch.cat(
            [reference_points, torch.ones_like(reference_points[..., :1])], -1)
        # -> (num_points_in_pillar, bs, sum(hw), 4)
        reference_points = reference_points.permute(1, 0, 2, 3)
        num_Z_anchors, bs, num_queries, _ = reference_points.shape
        num_cams = lidar2img.size(1)

        # -> (num_Z_anchors, bs, num_cams, num_queries, 4, 1)
        reference_points = reference_points.view(
            num_Z_anchors, bs, 1, num_queries, 4).repeat(1, 1, num_cams, 1, 1).unsqueeze(-1)

        # -> (num_Z_anchors, bs, num_cams, num_queries, 4, 4)
        lidar2img = lidar2img.view(
            1, bs, num_cams, 1, 4, 4).repeat(num_Z_anchors, 1, 1, num_queries, 1, 1)

        # convert to image coordinate
        # -> (num_Z_anchors, bs, num_cams, num_queries, 4)
        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)

    
        eps = 1e-5
        # filter z > 0 -> (num_Z_anchors, bs, num_cams, num_queries, 1)
        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = (reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps))

        # normalize u (x) / v (y) by image width / height
        # Prefer batch_input_shape (post-pad) if provided by data_preprocessor.
        meta0 = batch_data_samples[0].metainfo
        if 'batch_input_shape' in meta0 and meta0['batch_input_shape'] is not None:
            h = int(meta0['batch_input_shape'][0])
            w = int(meta0['batch_input_shape'][1])
        else:
            img_shape = meta0.get('img_shape', None)
            # multi-view: [(H, W, C), ...]
            if isinstance(img_shape, (list, tuple)) and len(img_shape) > 0 and isinstance(img_shape[0], (list, tuple)):
                h = int(img_shape[0][0])
                w = int(img_shape[0][1])
            # single-view: (H, W) or (H, W, C)
            elif isinstance(img_shape, (list, tuple)) and len(img_shape) >= 2 and isinstance(img_shape[0], (int, np.integer)):
                h = int(img_shape[0])
                w = int(img_shape[1])
            else:
                raise TypeError(f'Unsupported img_shape format in metainfo: {type(img_shape)} / {img_shape}')

        reference_points_cam[..., 0] /= w
        reference_points_cam[..., 1] /= h

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        
        bev_mask = torch.nan_to_num(bev_mask)
        # -> (num_cams, bs, num_queries, num_Z_anchors, 4)
        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        # -> (num_cams, bs, num_queries, num_Z_anchors)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask


    def forward(self,
                bev_query: Tensor,
                key: Optional[Tensor],
                value: Optional[Tensor],
                bev_h: int,
                bev_w: int,
                bev_pos: Optional[Tensor],
                spatial_shapes: Optional[Tensor],
                level_start_index: Optional[Tensor],
                valid_ratios: Optional[Tensor] = None,
                prev_bev: Optional[Tensor] = None,
                shift: Optional[Tensor] = None,
                batch_data_samples = None,
                **kwargs):
        """
        Args:
            bev_query: (bev_h*bev_w, bs, embed_dims)
            key: flattened_mlvl_feats, (num_cams, sum(HW), bs, embed_dims)
            value: flattened_mlvl_feats, (num_cams, sum(HW), bs, embed_dims)
            bev_pos: bev query positional embedding, (bev_h*bev_w, 1, embed_dims)

        """

        # enable fp16 where appropriate
        # (decorator kept separate for clarity)
        output = bev_query
        intermediate = []

        ref_3d = self.get_reference_points(
            bev_h, bev_w, self.pc_range[5] - self.pc_range[2],
            self.num_points_in_pillar,
            dim='3d',
            bs=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype)


        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d',
            bs=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype)
    

        
        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, batch_data_samples)
        
        shift_ref_2d = ref_2d.clone()
        if shift is not None:
            shift_ref_2d += shift[:, None, None, :]

        # debug prints removed

        # -> (bs, num_queries, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, num_queries, num_bev_level, _ = ref_2d.shape

        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = torch.stack([prev_bev, bev_query], 1).reshape(
                bs*2, num_queries, -1)
            hybrid_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs*2, num_queries, num_bev_level, 2)
        else:
            # [(bs, bev_h*bev_w, num_bev_levels, 2), (bs, bev_h*bev_w, num_bev_levels, 2)]
            # -> (bs, 2, bev_h*bev_w, num_bev_levels, 2)
            # -> (bs*2, bev_h*bev_w, num_bev_levels, 2)
            hybrid_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs*2, num_queries, num_bev_level, 2)

        # debug prints removed

        for lid, layer in enumerate(self.layers):
            output = layer(
                query=bev_query,
                key=key,
                value=value,
                bev_pos=bev_pos,
                ref_2d=hybrid_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                **kwargs)

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)
        
        if self.return_intermediate:
            return torch.stack(intermediate)
        
        return output


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
        """
        Args:
            query: bev_query, (bs, bev_h*bev_w, embed_dims)
            key: flattened_mlvl_img_feats, (num_cams, bs, sum(HW), embed_dims)
            value: flattened_mlvl_img_feats, (num_cams, bs, sum(HW), embed_dims)
            query_pos: here is None
            bev_pos: bev query positional embedding, (bev_h*bev_w, 1, embed_dims)
            reference_points_cam: bev_query anchors' projected points on multi view images,
                has shape (num_cams, bs, bev_h*bev_w, num_Z_anchors, 2)
            bev_mask: if reference_points_cam is valid,
                has shape (num_cams, bs, bev_h*bev_w, num_Z_anchors)
        """

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