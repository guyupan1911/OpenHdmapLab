import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import SGD

from mmengine.model import BaseModel
from mmengine.evaluator import BaseMetric
from mmengine.runner import Runner


# build a model
class MMResNet50(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()
    
    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels

model = MMResNet50()
# print(model)

# build a dataset and dataloader
norm_cfg = dict(mean=[0.491, 0.482, 0.447],
                std=[0.202, 0.199, 0.201])

train_dataset = torchvision.datasets.CIFAR10(
    'data/cifar-10-batches-py',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**norm_cfg)
]))
val_dataset = torchvision.datasets.CIFAR10(
    'data/cifar-10-batches-py',
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**norm_cfg)
]))
print(f'train size: {len(train_dataset)}')
print(f'val size: {len(val_dataset)}')

train_dataloader = DataLoader(batch_size=128,
                              shuffle=True,
                              dataset=train_dataset
)
val_dataloader = DataLoader(batch_size=128,
                            shuffle=True,
                            dataset=val_dataset
)

# build Evaluation Metrics
class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        return dict(accuarcy=100 * total_correct / total_size)


def data_preprocessor(data_batch):
    data = type(data_batch)(sample.to('cuda') for sample in data_batch)
    return data


# train
model.to('cuda')
max_epoch = 150
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
evaluator = Accuracy()
# run epoch
for epoch in range(max_epoch):
    # run iter
    model.train()
    for idx, data_batch in enumerate(train_dataloader):
        # data preprocessor
        data = data_preprocessor(data_batch)
        # run forward
        loss = model(*data, 'loss')['loss']
        # optim
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # val
    model.eval()
    with torch.no_grad():
        for idx, data_batch in enumerate(val_dataloader):
            data = data_preprocessor(data_batch)
            outputs = model(*data, 'predict')
            evaluator.process(data_batch=data_batch, data_samples=outputs)
        metrics = evaluator.evaluate(len(val_dataloader.dataset))
        print(f'epoch: {epoch}: {metrics}')

    # print(f'epoch: {epoch+1}, loss:{loss}')
        
# runner = Runner(
#     model = MMResNet50(),
#     work_dir='./work_dir',
#     train_dataloader=train_dataloader,
#     optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
#     train_cfg=dict(by_epoch=True, max_epochs=150, val_interval=1),
#     val_dataloader=val_dataloader,
#     val_cfg=dict(),
#     val_evaluator=dict(type=Accuracy),
# )

# runner.train()