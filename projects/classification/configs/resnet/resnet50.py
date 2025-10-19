from mmengine.config import read_base

with read_base():
    from .._base_.datasets.cifar10_bs512 import *
    from .._base_.models.resnet50 import *
    from .._base_.schedules.cifar10_bs512 import *
    from .._base_.default_runtime import *