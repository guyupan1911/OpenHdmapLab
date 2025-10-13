from typing import Optional, List, Sequence

from mmengine.model import BaseDataPreprocessor
from mmhdmap.registry import MODELS


@MODELS.register_module()
class ClsDataPreprocessor(BaseDataPreprocessor):

    def __init__(self,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Number = 0,
                 to_rgb: bool = False,
                 to_onehot: bool = False,
                 num_classes: Optional[int] = None,
                 batch_augments: Optional[dict] = None):
        super().__init__()
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        self.to_rgb = to_rgb
        self.to_onehot = to_onehot
        self.num_classes = num_classes

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                'preprocessing, please specify both `mean` and `std`.'
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

    def forward(self, data: dict, training: bool = False) -> dict:

        inputs = self.cast_data(data['inputs'])

        if isinstance(inputs, torch.Tensor):
            if self.to_rgb and inputs.size(1) == 3:
                inputs = inputs.flip(1)
            
            inputs = inputs.float()
            if self._enable_normalize:
                inputs = (inputs - self.mean) / self.std
            
            if self.pad_size_divisor > 1:
                h, w = inputs.shape[-2:]

                target_h = math.ceil(
                    h / self.pad_size_divisor) * self.pad_size_divisor
                target_w = math.ceil(
                    w / self.pad_size_divisor) * self.pad_size_divisor
                
                pad_h = target_h - h
                pad_w = target_w - w
                inputs = F.pad(inputs, (0, pad_w, 0, pad_h), 'constant',
                               self.pad_value)