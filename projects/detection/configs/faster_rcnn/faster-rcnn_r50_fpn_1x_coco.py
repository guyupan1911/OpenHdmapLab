from mmengine.config import read_base

with read_base():
    from .._base_.datasets.coco_detection import *
    from .._base_.models.faster_rcnn_r50_fpn import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *