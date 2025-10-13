from typing import Optional, List

import torch
import torch.nn.functional as F
import torchvision

from mmengine.model import BaseModel
from mmhdmap.registry import MODELS
from projects.classification.structures import DataSample


@MODELS.register_module()
class MMResNet50(BaseModel):
    def __init__(self,
                 data_preprocessor=None):
        super().__init__(data_preprocessor=data_preprocessor)
        self.resnet = torchvision.models.resnet50()

    def forward(self, inputs: torch.Tensor, data_samples: List[DataSample], mode: str):
        x = self.resnet(inputs)
        
        gt_labels = torch.stack([i.gt_label for i in data_samples]).reshape(-1)
        gt_labels = gt_labels.to(x.device)
        
        if mode == 'loss':    
            return {'loss': F.cross_entropy(x, gt_labels)}
        elif mode == 'predict':
            return x, gt_labels