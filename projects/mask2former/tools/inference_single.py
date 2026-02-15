import argparse
import os
import sys
from pathlib import Path

import cv2

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.dataset import Compose

from mmcv import imread

from mmdet.apis import inference_detector
from mmhdmap.registry import MODELS
from mmhdmap.registry import VISUALIZERS

from projects.mask2former.models.detectors import Mask2Map
from projects.mask2former.visualization import RotLocalVisualizer

DEFAULT_CLASSES = (
    "dotted lane", "solid lane", "curb", "fence", "crosswalk", "junction",
    "clear_area", "parking_space", "painted_island", "stop_line",
    "drivable_area", "sidewalk"
)
DEFAULT_PALETTE = [
    (255, 255, 255), (255, 255, 0), (255, 0, 255), (255, 0, 0), (0, 255, 0),
    (0, 255, 255), (0, 127, 255), (0, 255, 127), (0, 63, 255), (0, 255, 63),
    (0, 31, 255), (0, 255, 31)
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='inference single frame')
    parser.add_argument('--config', help='config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--image_path', help='lossless_image')
    parser.add_argument('--output_dir', help='output_dir')

    args = parser.parse_args()
    return args
    
def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    model = MODELS.build(cfg.model)

    load_checkpoint(model, args.checkpoint, map_location='cpu')

    model.cuda()
    model.eval()
    model.cfg = cfg

    rgb_image = imread(args.image_path, channel_order='rgb')

    test_pipeline = Compose([
        dict(type='mmdet.LoadImageFromNDArray', to_float32=True),
        dict(type='mmdet.PackDetInputs', meta_keys=('img_path', 'ori_shape', 'img_shape'))
    ])

    full_result = inference_detector(model, rgb_image, test_pipeline)

    
    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = {
        'classes': DEFAULT_CLASSES,
        'palette': DEFAULT_PALETTE
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, 'segmentation.png')
    if hasattr(full_result, 'pred_instances'):
        del full_result.pred_instances

    visualizer.add_datasample(
        'full_output',
        rgb_image,
        data_sample=full_result,
        draw_gt=False,
        draw_pred=True,
        show=False,
        wait_time=0,
        pred_score_thr=0.3,
        out_file=out_file,
    )
    print(f"Full output visualization saved to {out_file}")

if __name__ == '__main__':
    main()