from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR

# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=150, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=150,
        by_epoch=True,
        milestones=[100],
        gamma=0.1),
]

# optimizer
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(
                lr_mult=0.1,
                decay_mult=1.0))),
    clip_grad=dict(norm_type=2, max_norm=0.1))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)


