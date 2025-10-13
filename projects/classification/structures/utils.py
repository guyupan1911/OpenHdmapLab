from typing import List, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.utils import is_str

if hasattr(torch, 'tensor_split'):
    tensor_split = torch.tensor_split
else:
    def tensor_split(input: torch.Tensor, indices: list):
        outs = []
        for start, end in zip([0] + indices, indices + [input.size(0)]):
            outs.append(input[start:end])
        return outs


LABEL_TYPE = Union[torch.Tensor, np.ndarray, Sequence, int]
SCORE_TYPE = Union[torch.Tensor, np.ndarray, Sequence]


def format_label(value: LABEL_TYPE) -> torch.Tensor:
    if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim == 0:
        value = int(value.item())
    
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).to(torch.long)
    elif isinstance(value, Sequence) and not is_str(value):
        value = torch.tensor(value).to(torch.long)
    elif isinstance(value, int):
        value = torch.LongTensor([value])
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'Invalid type {type(value)} for format_label.')
    
    assert value.dim() == 1, \
        f'Label value should be a 1D tensor, but got {value.dim()}D tensor.'

    return value


def format_score(value: SCORE_TYPE) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).float()
    elif isinstance(value, Sequence) and not is_str(value):
        value = torch.tensor(value).float()
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'Invalid type {type(value)} for format_score.')
    assert value.dim() == 1, \
        f'Score value should be a 1D tensor, but got {value.dim()}D tensor.'

    return value


def cat_batch_labels(elements: List[torch.Tensor]):
    labels = []
    splits = [0]
    for element in elements:
        labels.append(element)
        splits.append(splits[-1] + element.size(0))
    batch_label = torch.cat(labels)
    return batch_label, splits[1:-1]


def batch_label_to_onehot(batch_label, split_indices, num_classes):
    sparse_onehot_list = F.one_hot(batch_label, num_classes)
    onehot_list = [
        sparse_onehot.sum(0)
        for sparse_onehot in tensor_split(sparse_onehot_list, split_indices)
    ]
    return torch.stack(onehot_list)


def label_to_onehot(label: LABEL_TYPE, num_classes: int):
    label = format_label(label)
    sparse_onehot = F.one_hot(label, num_classes)
    return sparse_onehot.sum(0)