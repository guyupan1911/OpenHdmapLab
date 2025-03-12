backbone=dict(
    type='ResNet',
    depth=101,
    num_stages=4,
    out_indices=(1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN2d', requires_grad=False),
    norm_eval=True,
    style='caffe',
    dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
    stage_with_dcn=(False, False, True, True))


