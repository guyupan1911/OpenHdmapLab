from mmdet.models.backbones import ResNet
from mmdet.models.necks import FPN
from projects.bevformer.models import (TemporalDet3DDataPreprocessor,
    BEVFormer, TemporalSelfAttention, SpatialCrossAttention,
    BEVFormerEncoder, BEVFormerEncoderLayer, BEVFormerDecoder, BEVFormerHead, NMSFreeCoder)

dim = 256
num_levels = 1

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]


model = dict(
    type=BEVFormer,
    bev_h=50,
    bev_w=50,
    num_query=900,
    video_test_mode=True,
    with_box_refine=True,
    pc_range=point_cloud_range,
    data_preprocessor=dict(
        type=TemporalDet3DDataPreprocessor,
        bgr_to_rgb=True,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        pad_size_divisor=32),
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
    bev_encoder=dict(
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
        )),
    decoder=dict(
        type=BEVFormerDecoder,
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            cross_attn_cfg=dict(
                embed_dims=256,
                num_levels=1,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=512,
                ffn_drop=0.1))),
    positional_encoding=dict(
        type='mmdet.LearnedPositionalEncoding',
        num_feats=128,
        row_num_embed=50,
        col_num_embed=50,
    ),
    bbox_head=dict(
        type=BEVFormerHead,
        bev_h=50,
        bev_w=50,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        bbox_coder=dict(
            type=NMSFreeCoder,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            num_classes=10))
)