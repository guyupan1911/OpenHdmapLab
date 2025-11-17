import argparse
import os.path as osp

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchsummary import summary
from PIL import Image

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.runner import load_checkpoint
from mmengine.dataset.sampler import DefaultSampler
from mmengine.dataset import default_collate

from mmdet.engine.hooks.utils import trigger_visualization_hook

from mmhdmap.registry import DATASETS, MODELS
from mmengine.visualization import Visualizer
from mmdet.visualization import DetLocalVisualizer

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDetection test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir', help='directory where painted images will be saved')
    parser.add_argument('--wait-time', type=float, default=2, help='wait time for key press')
    args = parser.parse_args()
    return args

args = parse_args()

cfg = Config.fromfile(args.config)

if args.work_dir is not None:
    cfg.work_dir = args.work_dir
elif cfg.get('work_dir', None) is None:
    cfg.work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])

cfg.load_from = args.checkpoint

if args.show:
    cfg = trigger_visualization_hook(cfg, args)

runner = Runner.from_cfg(cfg)

def main():
    visualizer = DetLocalVisualizer()
    model = runner.model
    dataloader = runner.test_dataloader
    # Ensure checkpoint is loaded when running manual loop instead of runner.test()
    if getattr(cfg, 'load_from', None):
        load_checkpoint(model, cfg.load_from, map_location='cpu')
    # set dataset meta for proper class names/palette if available
    if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'metainfo'):
        visualizer.dataset_meta = dataloader.dataset.metainfo
    with torch.no_grad():
        model.eval()
        for data_batch in dataloader:
            output = model.test_step(data_batch)
            if (output[0].pred_instances is None):
                continue
            image = np.array(Image.open(output[0].img_path).convert('RGB'))
            visualizer.add_datasample(
                osp.basename(output[0].img_path),
                image,
                output[0],
                show=True)
            break

def run_val():
    runner.val()


if __name__ == '__main__':
    # main()
    run_val()
  