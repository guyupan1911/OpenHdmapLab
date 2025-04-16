import argparse

from mmengine.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description='DETR3D')
    parser.add_argument('config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--mode', choices=['train', 'test'], default='test')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    print(f'cfg: {cfg}')


if __name__=='__main__':
    main()