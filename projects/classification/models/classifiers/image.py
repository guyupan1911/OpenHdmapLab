from typing import Optional, List

import torch
import torch.nn as nn

from mmhdmap.registry import MODELS
from mmpretrain.structures import DataSample
from .base import BaseClassifier


@MODELS.register_module()
class ImageClassifier(BaseClassifier):

    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 train_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        if pretrained is not None:
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        data_preprocessor = data_preprocessor or {}

        if isinstance(data_preprocessor, dict):
            data_preprocessor.setdefault('type', 'ClsDataPreprocessor')
            data_preprocessor.setdefault('batch_augments', train_cfg)
            data_preprocessor = MODELS.build(data_preprocessor)
        elif not isinstance(data_preprocessor, nn.Module):
            raise TypeError('data_preprocessor should be a `dict` or '
                            f'`nn.Module` instance, but got '
                            f'{type(data_preprocessor)}')

        super(ImageClassifier, self).__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        if not isinstance(backbone, nn.Module):
            backbone = MODELS.build(backbone)
        if neck is not None and not isinstance(neck, nn.Module):
            neck = MODELS.build(neck)
        if head is not None and not isinstance(head, nn.Module):
            head = MODELS.build(head)

        self.backbone = backbone
        self.neck = neck
        self.head = head

        # If the model needs to load pretrain weights from a third party,
        # the key can be modified with this hook
        if hasattr(self.backbone, '_checkpoint_filter'):
            self._register_load_state_dict_pre_hook(
                self.backbone._checkpoint_filter)
    
    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'tensor'):
        if mode == 'tensor':
            feats = self.extract_feat(inputs)
            return self.head(feats) if self.with_head else feats
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise ValueError(f'Invalid mode: {mode}')
        
    def extract_feat(self, inputs, stage='neck'):
        assert stage in ['backbone', 'neck', 'pre_logits'], \
            f'invalid stage: {stage}'
        
        x = self.backbone(inputs)

        if stage == 'backbone':
            return x
        
        if self.with_neck:
            x = self.neck(x)
        
        if stage == 'neck':
            return x
        
        assert self.with_head and hasattr(self.head, 'pre_logits'), \
            'head.pre_logits must be set'
        
        return self.head.pre_logits(x)
    
    def loss(self, inputs: torch.Tensor,
             data_samples: List[DataSample]) -> dict:
        
        feats = self.extract_feat(inputs)
        return self.head.loss(feats, data_samples)
    
    def predict(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                **kwargs) -> List[DataSample]:
        feats = self.extract_feat(inputs)
        return self.head.predict(feats, data_samples, **kwargs)


        
