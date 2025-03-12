import argparse

from mmengine.config import Config
from mmhdmap.registry import MODELS

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMHdmap test (and eval) a model')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)

    model = MODELS.build(cfg.backbone)

    print(model)

if __name__ == '__main__':
    main()
