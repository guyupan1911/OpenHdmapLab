from mmdet.models.backbones import ResNet
from mmdet.models.necks import FPN
from projects.bevformer.models import TemporalSelfAttention, SpatialCrossAttention, BEVFormerEncoder, BEVFormerEncoderLayer

dim = 256
num_levels = 1

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]


model = dict(
    img_backbone=dict(
        type=ResNet,
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type=FPN,
        in_channels=[2048],
        out_channels=dim,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=num_levels,
        relu_before_extra_convs=True),
    bevformer_encoder=dict(
        type=BEVFormerEncoder,
        num_layers=3,
        pc_range=point_cloud_range,
        num_points_in_pillar=4,
        return_intermediate=False,
        layer_cfg=dict(
            temporal_attn_cfg=dict(
                type=TemporalSelfAttention,
                embed_dims=256,
                num_levels=1,
                batch_first=True),
            spatial_cross_attn_cfg=dict(
                type=SpatialCrossAttention,
                embed_dims=256,
                attn_cfg=dict(
                    type='MSDeformableAttention3D',
                    embed_dims=256,
                    num_points=8,
                    num_levels=1,
                    batch_first=True)),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=512,
                ffn_drop=0.1)
        ))
)