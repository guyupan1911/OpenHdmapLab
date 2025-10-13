from torch import nn
from mmhdmap.registry import ACTIVATION


@ACTIVATION.register_module()
class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print("call Sigmoid.forward")
        return x


@ACTIVATION.register_module()
class ReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print("call ReLU.forward")
        return x