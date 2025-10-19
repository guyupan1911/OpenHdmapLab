import torch
import torch.nn as nn

from mmhdmap.registry import MODELS


@MODELS.register_module()
class GlobalAveragePooling(nn.Module):
    
    def __init__(self, dim=2):
        super().__init__()
        
        if dim == 1:
            self.gap = nn.AdaptiveAvgPool1d(1)
        elif dim == 2:
            self.gap = nn.AdaptiveAvgPool2d((1,1))
        elif dim == 3:
            self.gap = nn.AdaptiveAvgPool3d((1,1,1))
    
    def init_weights(self):
        pass
    
    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([self.gap(x) for x in inputs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)]
            )
        elif  isinstance(inputs, torch.Tensor):
            outs = self.gap(inputs)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise ValueError(f'Invalid type {type(inputs)} for forward.')
        
        return outs