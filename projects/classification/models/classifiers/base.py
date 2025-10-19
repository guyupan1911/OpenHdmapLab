from abc import ABCMeta, abstractmethod
from typing import Optional, List, Sequence

import torch

from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement


class BaseClassifier(BaseModel, metaclass=ABCMeta):

    def __init__(self,
                 init_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None):
        super(BaseClassifier, self).__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

    @property
    def with_neck(self) -> bool:
        return hasattr(self, 'neck') and self.neck is not None
    
    @property
    def with_head(self) -> bool:
        return hasattr(self, 'head') and self.head is not None
    
    @abstractmethod
    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'tensor'):
        pass

    def extract_feat(self, inputs: torch.Tensor):
        raise  NotImplementedError
    
    def extract_feats(self, multi_inputs: Sequence[torch.Tensor], **kwargs):
        assert isinstance(multi_inputs, Sequence), \
            f'multi_inputs must be a sequence, but got {type(multi_inputs)}'
        
        return [self.extract_feat(inputs, **kwargs) for inputs in multi_inputs]
    