import torch

from mmhdmap.registry import ACTIVATION
from mmhdmap.models import activations

print(ACTIVATION.module_dict)

input = torch.randn(2)

act_cfg = dict(type='ReLU')
activation = ACTIVATION.build(act_cfg)

print(activation(input))