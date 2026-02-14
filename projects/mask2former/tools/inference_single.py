import argparse
import sys
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import load_checkpoint

from mmhdmap.registry import MODELS

from projects.mask2former.models.detectors import Mask2Map

def parse_args():
    parser = argparse.ArgumentParser(
        description='inference single frame')
    parser.add_argument('--config', help='config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')

    args = parser.parse_args()
    return args
    

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    model = MODELS.build(cfg.model)

    load_checkpoint(model, args.checkpoint, map_location='cpu')


if __name__ == '__main__':
    main()