from typing import Tuple, List, Dict

import torch
from torch import Tensor

from mmcv.cnn.bricks.transformer import TransformerLayerSequence, BaseTransformerLayer

from mmhdmap.registry import MODELS


@MODELS.register_module()
class BEVFormerEncoder(TransformerLayerSequence):

    def __init__(self,
                 pc_range: Tuple[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 num_points_in_pillar: int = 4,
                 return_intermediate: bool = False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False

    @staticmethod
    def get_reference_points(H: int = 200,
                             W: int = 200,
                             Z: int = 8,
                             num_points_in_pillar: int = 4,
                             dim: str = '3d',
                             bs: int = 1,
                             device: str = 'cuda',
                             dtype: torch.dtype = torch.float) -> Tensor:
        
        if dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
            )

            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack([ref_x, ref_y], -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d
        
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                device=device).view(-1, 1, 1).expand(num_points_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                device=device).view(1, 1, W).expand(num_points_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                device=device).view(-1, H, 1).expand(num_points_pillar, H, W) / H
            ref_3d = torch.stack([xs, ys, zs], -1) # (num_points_pillar, H, W, 3)
            # -> (num_points_pillar, 3, H, W)
            # -> (num_points_pillar, 3, H*W)
            # -> (num_points_pillar, H*W, 3)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            # -> (bs, num_points_pillar, H*W, 3)
            ref_3d = ref3d[None].repeat(bs, 1, 1, 1)
            return ref_3d
    
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self,
                       reference_points: Tensor, #(bs, num_points_pillar, H*W, 3)
                       pc_range: Tuple[float],
                       img_metas: List[Dict]) -> Tuple[Tensor, Tensor]:
        
        # collect lidar2img matrix
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(imf_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img) #(B, num_cams, 4, 4)
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = (
            reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0])        
        reference_points[..., 1:2] = (
            reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1])
        reference_points[..., 2:3] = (
            reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2])
        
        # (bs, num_points_pillar, H*W, 4)
        reference_points = torch.cat(
            [reference_points, torch.ones_like(reference_points[..., :1])], -1)
        
        # -> (num_points_pillar, bs, H*W, 4)
        reference_points = reference_points.permute(1, 0, 2, 3)
        D, bs, num_queries = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        # -> (num_points_pillar, bs, num_cam, num_queries, 4, 1)
        reference_points = reference_points.view(
            D, bs, 1, num_queries, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
        
        # (num_points_pillar, bs, num_cam, num_query, 4, 4)
        lidar2img = lidar2img.view(
            1, bs, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        # -> (num_points_pillar, bs, num_cam, num_queries, 4)
        reference_points_cam = torch.matmul(
            lidar2img.to(torch.float32),
            reference_points.to(torch.float32)).squeeze(-1)
        
        eps = 1e-5

        # (num_points_pillar, bs, num_cam, num_queries, 1)
        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maxium(
            reference_points_cam[..., 2:3],
            torch.ones_like(reference_points_cam[..., 2:3]) * eps)
        
        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        
        bev_mask = torch.nan_to_num(bev_mask)

        # -> (num_cam, bs, num_queries, num_points_pillar, 4)
        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        # -> (num_cam, bs, num_queries, num_points_pillar, 1)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4)

        return reference_points_cam, bev_mask
    
    @auto_fp16()
    def forward(self,
                bev_query: Tensor,
                key: Optional[Tensor],
                value: Optional[Tensor],
                bev_h: int,
                bev_w: int,
                bev_pos: Optional[Tensor],
                spatial_shapes,
                level_start_index,
                valid_ratios,
                prev_bev: Optional[Tensor],
                shift,
                *args,
                **kwargs):
        output = bev_query
        intermediate = []

        ref_3d = self.get_reference_points(
            bev_h, bev_w, self.pc_range[5] - self.pc_range[2],
            self.num_points_in_pillar, dim='3d',
            bs = bev_query.size(1), device=bev_query.device,
            dtype=bev_query.dtype)
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d', bs=bev_query.size(1), device=bev_query.device,
            dtype=bev_query.dtype)
        
        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, kwargs['img_metas'])
        
        # (bs, H*W , 1, 2)
        shift_ref_2d = ref_2d.clone()
        shift_ref_2d += shift[:, None, None, :]

        # (bs, num_queries, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            # -> (bs, 2, num_queries, embed_dims]
            # -> (bs*2, num_queries, embed_dims)
            prev_bev = torch.stack(
                [prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)
            # -> (bs, 2, len_bev, 1, 2)
            # -> (bs*2, len_bev, 1, 2)
            hybrid_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)
        else:
            hybrid_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)
        
        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
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
                *args,
                **kwargs)
            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)
        
        if self.return_intermediate:
            return torch.stack(intermediate)
        
        return output


@MODELS.register_module()
class BEVFormerLayer(BaseTransformerLayer):

    def __init__(self,
                 attn_cfgs,
                 ffn_cfgs,
                 operation_order,
                 batch_first=True,
                 **kwargs):
        super().__init__(
            attn_cfgs=attn_cfgs,
            ffn_cfgs=ffn_cfgs,
            operation_order=operation_order,
            batch_first=batch_first,
            **kwargs)
        
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
    
    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shape=None,
                level_start_index=None,
                prev_bev=None,
                **kwargs):
        
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attm_masks) for _ in range(self.num_attn)
            ]
        else:
            assert len(attn_masks) == self.num_attn
        
        for layer in self.operation_order:
            if layer == 'self_attn':
                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_masks=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor(
                        [[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query
            elif layer == 'norm':
                query = self.norms[norm_index][query]
                norm_index += 1
            
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query
            
            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1
        
        return query