import os
from typing import Optional, List, Union, Callable

import pandas as pd

from mmengine.dataset import BaseDataset
from mmhdmap.registry import DATASETS
from .categories import CIFAR10_CATEGORIES

@DATASETS.register_module()
class CIFAR10(BaseDataset):
    METAINFO = {'classes': CIFAR10_CATEGORIES}

    def __init__(self,
                 ann_file: Optional[str] = None,
                 data_root: str = '',
                 data_prefix: dict = dict(img_path=''),
                 split: str = 'train',
                 metainfo: Optional[dict] = None,
                 serialize_data: bool = False,
                 mode: str = 'train',
                 pipeline: List[Union[dict, Callable]] = [],
                 **kwargs):
        self.mode = mode

        self.class_to_idx = {cat : i for i, cat in enumerate(self.METAINFO['classes'])}
        self.idx_to_class = {i : cat for i, cat in enumerate(self.METAINFO['classes'])}

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            serialize_data=serialize_data,
            pipeline=pipeline,
            **kwargs)
    
    def get_idx_to_class(self):
        return self.idx_to_class

    def load_data_list(self):
        ann_file = self.ann_file
        df = pd.read_csv(ann_file)

        ids = df['id'].tolist()
        labels = df['label'].tolist()
        
        data_list = []
        img_prefix = os.path.join(self.data_prefix['img_path'], self.mode)
        for id, label in zip(ids, labels):
            info = {'img_path': os.path.join(img_prefix, str(id) + '.png'),
                    'gt_label': int(self.class_to_idx[label])}
            data_list.append(info)
        
        return data_list
        



