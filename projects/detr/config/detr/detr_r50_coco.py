from mmengine.config import read_base

with read_base():
    from .._base_.datasets.coco_detection import *

