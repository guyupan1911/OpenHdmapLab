default_scope = 'mmhdmap'

model=dict(
    type='MSDeformAttnPixelDecoder',
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
        init_cfg=None,
)

