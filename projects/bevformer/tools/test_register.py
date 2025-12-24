from mmhdmap.registry import MODELS

from mmengine.config import Config

cfg = Config.fromfile('projects/bevformer/configs/bevformer/bevformer.py')


data_preprocessor=dict(
        type='TemporalDet3DDataPreprocessor',
        bgr_to_rgb=False,
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        pad_size_divisor=32)


bev_encoder=dict(
    type='BEVFormerEncoder',
    num_layers=6,
    pc_range=None,
    num_points_in_pillar=4,
    return_intermediate=False,
    bev_h=200,
    bev_w=200,
    positional_encoding=dict(
        type='mmdet.LearnedPositionalEncoding',
        num_feats=128,
        row_num_embed=200,
        col_num_embed=200,
    ),
    layer_cfg=dict(
        temporal_attn_cfg=dict(
            type='TemporalSelfAttention',
            embed_dims=256,
            num_levels=1,
            batch_first=True),
        spatial_cross_attn_cfg=dict(
            type='SpatialCrossAttention',
            embed_dims=256,
            attn_cfg=dict(
                type='MSDeformableAttention3D',
                embed_dims=256,
                num_points=8,
                num_levels=4,
                batch_first=True)),
        ffn_cfg=dict(
            embed_dims=256,
            feedforward_channels=512,
            ffn_drop=0.1)
    ))

bev_encoder = MODELS.build(bev_encoder)

# data_preprocessor = MODELS.build(data_preprocessor)