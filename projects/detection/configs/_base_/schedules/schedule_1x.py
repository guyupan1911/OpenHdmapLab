from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR

# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,
        by_epoch=True,
        milestones=[40],
        gamma=0.1),
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(lr=0.0002, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1),
            reference_points=dict(lr_mult=0.1),
            sampling_offsets=dict(lr_mult=0.1))),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    )

auto_scale_lr = dict(base_batch_size=32)
