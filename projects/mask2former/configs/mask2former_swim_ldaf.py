default_scope = 'mmhdmap'


model = dict(
    type='Mask2Map',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[60.62874883485384]*3, std=[45.485172265177695]*3, # non-zero
        # mean=[19.789781675741573]*3, std=[31.705467057439492]*3, # total
        bgr_to_rgb=True,
        pad_size_divisor=32,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False,
        boxtype2tensor=False,
        batch_augments=None),
    backbone=dict(
        # _delete_=True,
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths = [2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        frozen_stages=-1,
        init_cfg=None,
    ),
    neck=None,
    bev_encoder=None,
    bev_neck=dict(
        type='MSDeformAttnPixelDecoder',
        in_channels=[96, 192, 384, 768],
        feat_channels=256,
        out_channels=256,
        num_outs=3,
        norm_cfg=dict(type='GN', num_groups=32),
        act_cfg=dict(type='ReLU'),
        encoder=dict(
            type='DetrTransformerEncoder',
            num_layers=6,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=dict(
                    type='MultiScaleDeformableAttention',
                    embed_dims=256,
                    num_heads=8,
                    num_levels=3,
                    num_points=4,
                    im2col_step=64,
                    dropout=0.0,
                    batch_first=False,
                    norm_cfg=None,
                    init_cfg=None),
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=256,
                    feedforward_channels=1024,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True)),
                operation_order=('self_attn', 'norm', 'ffn', 'norm')),
            init_cfg=None),
            positional_encoding=dict(
                type='mmdet.SinePositionalEncoding', num_feats=128, normalize=True),
            init_cfg=None
    ),
    mask_decoder=dict(
        type="DetrTransformerDecoder",
        num_layers=9,
        return_intermediate=True,
        transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=256,
                    # feedforward_channels=_ffn_dim_,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                # feedforward_channels=_ffn_dim_,
                feedforward_channels=2048,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                'ffn', 'norm')),
            init_cfg=None
    ),
    mask_head=dict(
        type='Mask2FormerHead',
        feat_channels=256,
        out_channels=256,
        num_things_classes=11,
        num_stuff_classes=1,
        num_queries=400,
        num_transformer_feat_level=3,
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * 11 + [0.1]),
        loss_mask=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='mmdet.DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        train_cfg=dict(
            assigner=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=2.0),
                    dict(type='mmdet.CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                    dict(type='mmdet.DiceCost', weight=5.0, pred_act=True, eps=1.0)
                ]),
            sampler=dict(type='mmdet.MaskPseudoSampler')),
        test_cfg=dict(max_per_img=50),
    ),
    panoptic_fusion_head=dict(
        type='MaskFormerFusionHead',
        num_things_classes=11,
        num_stuff_classes=1,
        loss_panoptic=None,
        init_cfg=None,
        test_cfg=dict(
            panoptic_on=True,
            # For now, the dataset does not support
            # evaluating semantic segmentation metric.
            semantic_on=False,
            instance_on=True,
            # max_per_image is for instance segmentation.
            max_per_image=200,
            iou_thr=0.8,
            # In Mask2Former's panoptic postprocessing,
            # it will filter mask area where score is less than 0.5 .
            filter_low_score=True)
    ),
    map_decoder=None,
    map_head=None,

    # bbox_decoder=_bbox_decoder_cfg_,
    # bbox_head=_bbox_head_cfg_,

    # ldaf_head=_ldaf_head_cfg_,
    # ldaf_postprocessor=_ldaf_postprocessor_,

    map_query_generator=None,
    map_denoiser=None,

    positional_encoding=dict(
        type="mmdet.SinePositionalEncoding",
        num_feats=256 // 2,
        normalize=True
    ),

    num_queries=400,
    num_transformer_feat_level=3,

    train_cfg=dict(
        task_mode='multi_modal',
    ),
    
    test_cfg=dict(
        inference_tasks=['mask', 'bbox'],
    ),

    frozen_stages=['backbone', 'bev_neck', 'mask_stage'],
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)