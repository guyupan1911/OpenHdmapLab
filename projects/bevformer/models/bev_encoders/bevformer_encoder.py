import copy 
from typing import Optional, List, Tuple, Sequence

import torch
from torch import Tensor
from torch import nn
import numpy as np

from mmengine.model import BaseModule, ModuleList
from mmengine.config import ConfigDict

from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN
from mmdet.utils import OptConfigType, ConfigType, OptMultiConfig

from mmhdmap.registry import MODELS



@MODELS.register_module()
class BEVFormerEncoder(BaseModule):

    def __init__(self, 
                 pc_range: Tuple[int],
                 num_feature_levels: int = 4,
                 num_points_in_pillar: int = 4,
                 return_intermediate: bool = False,
                 num_layers: int = 6,
                 num_cp: int = -1,
                 num_cams: int = 6,
                 use_cams_embeds: bool = True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 bev_h: int = 50,
                 bev_w: int = 50,
                 positional_encoding: OptMultiConfig = None,
                 rotate_center: Optional[Sequence[float]] = None,
                 layer_cfg: Optional[ConfigDict] = None,
                 init_cfg: Optional[ConfigDict] = None) -> None:

        super().__init__(init_cfg=init_cfg)
        self.pc_range = pc_range
        self.num_feature_levels = num_feature_levels
        self.num_points_in_pillar = num_points_in_pillar
        self.return_intermediate = return_intermediate
        self.num_layers = num_layers
        self.layer_cfg = layer_cfg
        self.num_cp = num_cp
        assert self.num_cp <= self.num_layers

        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.num_cams = num_cams
        self.use_cams_embeds = use_cams_embeds
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.positional_encoding = positional_encoding

        # torchvision.transforms.functional.rotate expects center=(x, y) in pixel coords.
        # If not provided, use the BEV feature map center for stability across bev_h/bev_w.
        if rotate_center is None:
            self.rotate_center = ((self.bev_w - 1) / 2.0, (self.bev_h - 1) / 2.0)
        else:
            assert len(rotate_center) == 2, 'rotate_center must be a 2-tuple/list: (x, y)'
            self.rotate_center = (float(rotate_center[0]), float(rotate_center[1]))

        self._init_layers()
    
    def _init_layers(self) -> None:
        self.layers = ModuleList([
            BEVFormerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])

        self.embed_dims = self.layers[0].embed_dims
        self.bev_embedding = nn.Embedding(
                        self.bev_h * self.bev_w, self.embed_dims)

        self.positional_encoding = MODELS.build(self.positional_encoding)


        self.level_embeds = nn.Parameter(torch.Tensor(
        self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))

        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True)
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))
    
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

    def point_sampling(self, reference_points, pc_range, batch_data_samples):
        """
        Args:
            reference_points: 3d points on bev plane, has shape
                (bs, num_points_in_pillar, sum(HW), 3)
            pc_range: lidar range
            batch_data_samples: metainfo
        """
        # with torch.cuda.amp.autocast(enabled=False):
        lidar2img = []
        for data_sample in batch_data_samples:
            lidar2img.append(copy.deepcopy(data_sample.metainfo['lidar2img']))
        lidar2img = np.asarray(lidar2img)
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


    def prepare_encoder(self, mlvl_feats, batch_data_samples, prev_bev):
    
        bs, num_cams, _, _, _ = mlvl_feats[0].shape
        
        # prepare bev embed
        bev_query = self.bev_embedding.weight
        bev_mask = bev_query.new_zeros((bs, self.bev_h, self.bev_w))
        bev_pos = self.positional_encoding(bev_mask)

        bev_query = bev_query.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # align prev_bev
        # obtain rotation angle and shift with ego motion
        delta_x = np.array([each.metainfo['can_bus'][0] for each in batch_data_samples])
        delta_y = np.array([each.metainfo['can_bus'][1] for each in batch_data_samples])
        # BEVFormer can_bus convention:
        # - can_bus[-2]: absolute ego yaw in radians (used for shift direction)
        # - can_bus[-1]: delta ego yaw in degrees (used for rotating prev_bev)
        ego_angle = np.array(
            [each.metainfo['can_bus'][-2] / np.pi * 180 for each in batch_data_samples])

        # pc_range = [xmin, ymin, zmin, xmax, ymax, zmax]
        grid_length_y = (self.pc_range[4] - self.pc_range[1]) / self.bev_h
        grid_length_x = (self.pc_range[3] - self.pc_range[0]) / self.bev_w
        translation_length = np.sqrt(delta_x**2 + delta_y**2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / self.bev_h
        shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / self.bev_w

        shift = bev_query.new_tensor(
            [shift_x, shift_y]).permute(1, 0)

        if prev_bev is not None:
            if prev_bev.shape[1] == self.bev_h * self.bev_w:
                # -> (bev_h*bev_w, bs, embed_dims)
                prev_bev = prev_bev.permute(1, 0, 2)
            for i in range(bs):
                rotation_angle = batch_data_samples[i].metainfo['can_bus'][-1]
                tmp_prev_bev = prev_bev[:, i].reshape(
                    self.bev_h, self.bev_w, -1).permute(2, 0, 1)
                # Guard against a misconfigured rotate_center (must lie within the BEV map).
                if not (0.0 <= self.rotate_center[0] <= (self.bev_w - 1) and
                        0.0 <= self.rotate_center[1] <= (self.bev_h - 1)):
                    raise ValueError(
                        f'rotate_center={self.rotate_center} is outside the BEV map '
                        f'(bev_w={self.bev_w}, bev_h={self.bev_h}). '
                        'Set rotate_center=None to use the BEV center, or pass a valid (x, y).'
                    )
                tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                      center=self.rotate_center)
                tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                    self.bev_h * self.bev_w, 1, -1)
                prev_bev[:, i] = tmp_prev_bev[:, 0]

        # use can bus
        can_bus = bev_query.new_tensor(
            [each.metainfo['can_bus'] for each in batch_data_samples])
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]

        bev_query = bev_query + self.use_can_bus * can_bus

        #
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cams, C, H, W = feat.shape
            spatial_shape = (H, W)
            # -> (num_cams, bs, H*W, C)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :]
            feat = feat + self.level_embeds[None, None, lvl:lvl+1, :]
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
        
        # -> (num_cams, sum(HW), bs, embed_dims)
        feat_flatten = torch.cat(feat_flatten, 2)
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)

        # -> (num_levels, 2）
        spatial_shapes = bev_pos.new_tensor(spatial_shapes, dtype=torch.long)
        level_sizes = spatial_shapes.prod(dim=1)
        level_start_index = torch.cat([level_sizes.new_zeros(1), level_sizes.cumsum(0)[:-1]])

        ref_3d = self.get_reference_points(
            self.bev_h, self.bev_w, self.pc_range[5] - self.pc_range[2],
            self.num_points_in_pillar,
            dim='3d',
            bs=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype)


        ref_2d = self.get_reference_points(
            self.bev_h, self.bev_w, dim='2d',
            bs=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype)
    
       
        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, batch_data_samples)
        
        shift_ref_2d = ref_2d.clone()
        if shift is not None:
            shift_ref_2d += shift[:, None, None, :]
        

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


        return bev_query, bev_pos, hybrid_ref_2d, feat_flatten, spatial_shapes, level_start_index, bev_mask, reference_points_cam


    def forward(self, mlvl_feats, batch_data_samples, prev_bev) -> Tensor:
        """
        Args:

        Returns:
            bev_embedding: bev features after temporal self attn and spatial cross attn,
                has shape (bs, bev_h*bev_w, embed_dims)
        """

        outputs = self.prepare_encoder(mlvl_feats, batch_data_samples, prev_bev)
        bev_query, bev_pos, hybrid_ref_2d, feat_flatten, spatial_shapes, level_start_index, bev_mask,
            reference_points_cam = outputs
    
        output = bev_query
        intermediate = []

        for lid, layer in enumerate(self.layers):
            output = layer(
                query=bev_query,
                key=feat_flatten,
                value=feat_flatten,
                bev_pos=bev_pos,
                ref_2d=hybrid_ref_2d,
                bev_h=self.bev_h,
                bev_w=self.bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev)

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)
        
        if self.return_intermediate:
            return torch.stack(intermediate)
        
        return {
            "bev_embed": output
        }


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
        self.temporal_attn = MODELS.build(self.temporal_attn_cfg)
        self.embed_dims = self.temporal_attn.embed_dims
        self.spatial_cross_attn = MODELS.build(self.spatial_cross_attn_cfg)
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