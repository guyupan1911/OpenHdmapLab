from mmdet.models.backbones import ResNet
from mmdet.models.necks import FPN

dim = 256
num_levels = 1

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
)