import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

from mmcv.transforms import BaseTransform

from mmhdmap.registry import TRANSFORMS
from projects.classification.structures import DataSample


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(
            f'Type {type(data)} cannot be converted to tensor.'
            'Supported types are: `numpy.ndarray`, `torch.Tensor`, '
            '`Sequence`, `int` and `float`')


@TRANSFORMS.register_module()
class PackInputs(BaseTransform):

    DEFAULT_META_KEYS = ('sample_idx', 'img_path', 'ori_shape', 'img_shape',
                     'scale_factor', 'flip', 'flip_direction')

    def __init__(self,
                 input_key='img',
                 algorithm_keys=(),
                 meta_keys=DEFAULT_META_KEYS):
        self.input_key = input_key
        self.algorithm_keys = algorithm_keys
        self.meta_keys = meta_keys

    @staticmethod
    def format_input(input_):
        if isinstance(input_, list):
            return [PackInputs.format_input(item) for item in input_]
        elif isinstance(input_, np.ndarray):
            if input_.ndim == 2:
                input_ = np.expand_dims(input_, -1)
            if input_.ndim == 3 and not input_.flags.c_contiguous:
                input_ = np.ascontiguousarray(input_.transpose(2, 0, 1))
                input_ = to_tensor(input_)
            elif input_.ndim == 3:
                input_ = to_tensor(input_).permute(2, 0, 1).contiguous()
            else:
                input_ = to_tensor(input_)
        elif isinstance(input_, Image.Image):
            input_ = F.pil_to_tensor(input_)
        elif not isinstance(input_, torch.Tensor):
            raise TypeError(f'Invalid type {type(input_)} for format_input.')
        
        return input_

    def transform(self, results: dict) -> dict:
        packed_results = dict()
        if self.input_key in results:
            input_ = results[self.input_key]
            packed_results['inputs'] = self.format_input(input_)

        data_sample = DataSample()


        if 'gt_label' in results:
            data_sample.set_gt_label(results['gt_label'])
        if 'gt_score' in results:
            data_sample.set_gt_score(results['gt_score'])
        if 'mask' in results:
            data_sample.set_mask(results['mask'])

        for key in self.algorithm_keys:
            if key in results:
                data_sample.set_field(results[key], key)
        
        for key in self.meta_keys:
            if key in results:
                data_sample.set_field(results[key], key, field_type='metainfo')

        packed_results['data_samples'] = data_sample
        return packed_results
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(input_key='{self.input_key}', "
        repr_str += f'algorithm_keys={self.algorithm_keys}, '
        repr_str += f'meta_keys={self.meta_keys})'
        return repr_str