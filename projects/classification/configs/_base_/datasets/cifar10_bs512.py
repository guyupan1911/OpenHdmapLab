from mmpretrain.datasets import RandomCrop, RandomFlip, PackInputs
from mmpretrain.models.utils import ClsDataPreprocessor
from projects.classification.dataset import CIFAR10, LoadImageFromFile
from mmpretrain.evaluation import Accuracy


data_preprocessor = dict(
    type=ClsDataPreprocessor,
    num_classes=10,
    # RGB format normalization parameters
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    # loaded images are already RGB format
    to_rgb=True)

train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=RandomCrop, crop_size=32, padding=4),
    dict(type=RandomFlip, prob=0.5, direction='horizontal'),
    dict(type=PackInputs),
]

train_dataset = dict(
    type=CIFAR10,
    data_root='data/cifar-10',
    ann_file='trainLabels.csv',
    pipeline=train_pipeline)

test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=PackInputs),
]

test_dataset = dict(
    type=CIFAR10,
    data_root='data/cifar-10',
    ann_file='trainLabels.csv',
    pipeline=test_pipeline)

train_dataloader = dict(
    batch_size=512,
    num_workers=4,
    dataset=train_dataset,
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=512,
    num_workers=4,
    dataset=test_dataset,
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = dict(
    type=Accuracy,
)

val_cfg = dict()