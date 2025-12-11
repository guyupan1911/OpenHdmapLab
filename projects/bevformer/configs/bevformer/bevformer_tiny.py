from mmengine.config import read_base

with read_base():
    from .._base_.datasets.nuscenes_temporal import *
    from .._base_.models.bevformer_tiny import *
    # from .._base_.schedules.schedule_1x import *
    # from .._base_.default_runtime import *