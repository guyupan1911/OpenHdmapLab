import argparse

import torch
from torch.utils.data import DataLoader

from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model


from datasets import CustomNuScenesDataset
from visualization import visualization_data_sample
from models import BEVFormer

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
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes')

    # build model
    model = BEVFormer(**cfg.model)
    model.cuda()
    model.eval()
    
    # load checkpoint
    checkpoint = _load_checkpoint(args.checkpoint)
    checkpoint = _load_checkpoint_to_model(model, checkpoint, strict=True)


    with torch.no_grad():
        for idx, data_batch in enumerate(dataloader):
            img = data_batch['img'].cuda()
            img_metas = data_batch['img_metas']
            # visualization_data_sample(nusc, data_batch)
            # model inference
            results = model(img, img_metas, False)

            return
        
if __name__=='__main__':
    main()