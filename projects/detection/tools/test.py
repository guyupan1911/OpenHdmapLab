import argparse
import os.path as osp

from mmengine.config import Config
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook

from mmhdmap.registry import DATASETS

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

def main():
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
    runner.test()
   
def test_modules():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    dataset = DATASETS.build(cfg.val_dataloader.dataset)

    print(f'dataset size: {len(dataset)}')
    # print(dataset[0]['inputs'].shape)
    # print(dataset[0]['data_samples'])
    print(dataset[0])


if __name__ == '__main__':
    main()
    # test_modules()