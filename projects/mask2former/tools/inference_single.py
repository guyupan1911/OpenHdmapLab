import argparse
import sys
from pathlib import Path

from mmengine.config import Config

from mmhdmap.registry import MODELS

from projects.mask2former.models.layers import MSDeformAttnPixelDecoder, DetrTransformerDecoder

from projects.mask2former.models.seg_heads import MaskFormerFusionHead

def parse_args():
    parser = argparse.ArgumentParser(
        description='inference single frame')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')

    args = parser.parse_args()
    return args
    

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    model = MODELS.build(cfg.model)

    print(model)


if __name__ == '__main__':
    main()