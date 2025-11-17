from mmengine.config import read_base

with read_base():
    from .._base_.datasets.coco_detr import *
    from .._base_.models.detr import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *