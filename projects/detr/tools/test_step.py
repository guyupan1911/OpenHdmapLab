import argparse

import torch
from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint
import mmcv

from mmdet.utils import register_all_modules
from mmdet.registry import MODELS
from mmdet.visualization import DetLocalVisualizer



def parse_args():
    parser = argparse.ArgumentParser(description = 'test detr step bny step')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    # parser.add_argument('mode', help='train, val or evaluator')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    register_all_modules(init_default_scope=True)

    ## build data loader
    val_loader = Runner.build_dataloader(cfg.val_dataloader)

    ## build model
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.cuda()
    model.eval()

    ## visualize one sample and gt
    visualizer = DetLocalVisualizer(name='visualizer')
    visualizer.dataset_meta = val_loader.dataset.metainfo

    for idx, data_batch in enumerate(val_loader):

        with torch.no_grad():
            data_sample = model.test_step(data_batch)[0]

        print('num pred:', len(data_sample.pred_instances.bboxes))
        print('scores:', data_sample.pred_instances.scores[:10])

        img = mmcv.imread(data_sample.img_path, channel_order='rgb')

        visualizer.add_datasample(
            name = 'sample_with_gt',
            image = img,
            data_sample = data_sample,
            draw_gt=True,
            draw_pred=False,
            show=False,
            out_file='work_dirs/test_step/sample_with_gt.jpg'
        )

        visualizer.add_datasample(
            name = 'sample_with_gt',
            image = img,
            data_sample = data_sample,
            draw_gt=False,
            draw_pred=True,
            show=False,
            out_file='work_dirs/test_step/sample_with_pred.jpg'
        )
        break


if __name__ == '__main__':
    main()