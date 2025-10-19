from typing import Optional, Tuple

import torch
import torch.nn as nn

from mmhdmap.registry import MODELS
from mmpretrain.models.heads import ClsHead


@MODELS.register_module()
class LinearHead(ClsHead):

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 init_cfg: Optional[dict] = dict(
                    type='Normal', layer='Linear', std=0.01),
                 **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.fc = nn.Linear(in_channels, num_classes)
    
    def pre_logits(self, feats):
        return feats[-1]
    
    def forward(self, feats):
        pre_logits = self.pre_logits(feats)
        cls_score = self.fc(pre_logits)
        return cls_score