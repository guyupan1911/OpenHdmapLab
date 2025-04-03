import argparse

import torch
from torch.utils.data import DataLoader

import mmcv
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model
from mmengine.fileio import get

from mmdet3d.registry import DATASETS, MODELS, VISUALIZERS
from mmdet3d.visualization import Det3DLocalVisualizer
from detr3d.detr3d import DETR3D

def parse_args():
    parser = argparse.ArgumentParser(description='DETR3D')
    parser.add_argument('config')
    parser.add_argument('--checkpoint')

    args = parser.parse_args()
    return args


def visualization(visualizer, data_sample):
    # print(f'data_sample: {data_sample}')
    data_input=dict()
    assert 'img_path' in data_sample
    img_path = data_sample.img_path
    if isinstance(img_path, list):
        img = []
        for single_img_path in img_path:
            img_bytes = get(single_img_path)
            single_img = mmcv.imfrombytes(
                img_bytes, channel_order='rgb')
            img.append(single_img)
    else:
        img_bytes = get(img_path)
        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')

    data_input['img'] = img

    visualizer.add_datasample(
        'val_sample',
        data_input,
        data_sample=data_sample,
        draw_gt=False,
        draw_pred=True,
        show=True,
        vis_task='mono_det',
        pred_score_thr=0.3,
    )


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    # print(f'cfg: {cfg}')


    # build dataset
    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    dataloader = DataLoader(dataset, batch_size=5, collate_fn=pseudo_collate)

    # build model
    model = DETR3D(**cfg.model)
    model.to('cuda')
    model.eval()
    # print(model)    

    # load checkpoint
    checkpoint = _load_checkpoint(args.checkpoint)
    checkpoint = _load_checkpoint_to_model(model, checkpoint, strict=True)

    # visulizer
    # visualizer_cfg = cfg.visualizer
    # visualizer_cfg.setdefault('name', 'DETR3D')
    # visualizer_cfg.setdefault('save_dir', 'results')
    # visualizer = VISUALIZERS.build(visualizer_cfg)
    visualizer = Det3DLocalVisualizer()

    with torch.no_grad():
        for idx, data_batch in enumerate(dataloader):
            results = model.val_step(data_batch)
            visualization(visualizer, results[0])
            # return

if __name__=='__main__':
    main()