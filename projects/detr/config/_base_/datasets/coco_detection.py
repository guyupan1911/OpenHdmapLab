from mmcv.transforms import LoadImageFromFile

from mmengine.dataset.sampler import DefaultSampler

from mmdet.datasets import CocoDataset
from mmdet.datasets.transforms import (LoadAnnotations, PackDetInputs,
                                       RandomFlip, Resize)

dataset_type = CocoDataset
data_root = '/workspace/data/gpu/datasets/coco/'

backend_args = None

test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=Resize, scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))