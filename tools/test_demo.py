import argparse

from torch.utils.data import DataLoader

from mmengine.config import Config

from mmengine.registry import DATA_SAMPLERS, FUNCTIONS
from mmdet.registry import DATASETS


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMHdmap test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()    

    cfg = Config.fromfile(args.config)

    # create dataloader
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    print(f'dataset size: {len(dataset)}')

    sampler = DATA_SAMPLERS.build(
        cfg.test_dataloader.sampler,
        default_args=dict(dataset=dataset))

    collate = FUNCTIONS.get('pseudo_collate')

    dataloader = DataLoader(
        dataset = dataset,
        sampler = sampler,
        collate_fn = collate,
        worker_init_fn = None,
        batch_size=1,
        num_workers=2,
        persistent_workers=True,
        drop_last=False
    )

    for data_sample in dataloader:
        print(data_sample)
        break


if __name__ == '__main__':
    main()