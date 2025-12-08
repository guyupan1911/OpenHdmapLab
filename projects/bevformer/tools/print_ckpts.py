import torch

from collections import defaultdict

ckpt = torch.load('data/checkpoints/bevformer_tiny/bevformer_tiny_epoch_24.pth',
                  map_location='cpu')

print(ckpt.keys())
# print(ckpt['meta'])
# print(ckpt['optimizer'])

modules = defaultdict(list)
for key in ckpt['state_dict'].keys():
    module_name = '.'.join(key.split('.'))[:-1]
    modules[module_name].append(key)

for module_name, keys in modules.items():
    print(f'{module_name}:')
    for k in keys:
        if 'transformer' in k.lower():
            print(f' {k}')
    print()

