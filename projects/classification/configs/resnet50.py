# custom_imports = dict(imports=['projects.classification.models.resnet'], allow_failed_imports=False)
from torch.optim import SGD
from projects.classification.models import MMResNet50
from projects.classification.dataset import CIFAR10, LoadImageFromFile, PackInputs
from projects.classification.evaluation import Accuracy

work_dir = 'work_dir/resnet50'

model = dict(type=MMResNet50)

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),
)

train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=1)

data_preprocessor = dict(
    type='mmpretrain.ClsDataPreprocessor',
    # RGB format normalization parameters
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255])

train_pipeline = [
    dict(type=LoadImageFromFile),
    # dict(type='RandomCrop', size=32, padding=4),
    # dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type=PackInputs),
] 

dataset = dict(
    type=CIFAR10,
    data_root='data/cifar-10',
    ann_file='trainLabels.csv',
    pipeline=train_pipeline)

train_dataloader = dict(
    batch_size=128,
    num_workers=4,
    dataset=dataset,
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=128,
    num_workers=4,
    dataset=dataset,
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = dict(
    type=Accuracy,
)

val_cfg = dict()