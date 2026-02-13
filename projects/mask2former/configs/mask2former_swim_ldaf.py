default_scope = 'mmhdmap'

# model=dict(
#     type='MSDeformAttnPixelDecoder',
#     num_outs=3,
#     norm_cfg=dict(type='GN', num_groups=32),
#     act_cfg=dict(type='ReLU'),
#     encoder=dict(
#         type='DetrTransformerEncoder',
#         num_layers=6,
#         transformerlayers=dict(
#             type='BaseTransformerLayer',
#             attn_cfgs=dict(
#                 type='MultiScaleDeformableAttention',
#                 embed_dims=256,
#                 num_heads=8,
#                 num_levels=3,
#                 num_points=4,
#                 im2col_step=64,
#                 dropout=0.0,
#                 batch_first=False,
#                 norm_cfg=None,
#                 init_cfg=None),
#             ffn_cfgs=dict(
#                 type='FFN',
#                 embed_dims=256,
#                 feedforward_channels=1024,
#                 num_fcs=2,
#                 ffn_drop=0.0,
#                 act_cfg=dict(type='ReLU', inplace=True)),
#             operation_order=('self_attn', 'norm', 'ffn', 'norm')),
#         init_cfg=None),
#         positional_encoding=dict(
#             type='mmdet.SinePositionalEncoding', num_feats=128, normalize=True),
#         init_cfg=None,
# )

model = dict(
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
)

