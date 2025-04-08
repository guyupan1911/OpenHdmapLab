import argparse

import torch
from torch.utils.data import DataLoader

from mmengine.config import Config

from datasets import CustomNuScenesDataset
from visualization import visualization_data_sample

def parse_args():
    parser = argparse.ArgumentParser(description='DETR3D')
    parser.add_argument('config')
    parser.add_argument('--checkpoint')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    # print(f'cfg: {cfg.data.test}')

    # build dataset
    dataset = CustomNuScenesDataset(**cfg.data.val)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=True)

    for idx, data_batch in enumerate(dataloader):
        visualization_data_sample(nusc, data_batch)
        

if __name__=='__main__':
    main()