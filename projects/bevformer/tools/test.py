import argparse
import os
import os.path as osp
import copy

import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rich import print

from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint
from mmengine.dataset.utils import pseudo_collate
from mmdet3d.visualization import Det3DLocalVisualizer
from projects.bevformer.visualization import MultiFrameDet3DLocalVisualizer
import mmcv

from mmhdmap.registry import MODELS, DATASETS

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Bevformer test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir', help='directory where painted images will be saved')
    parser.add_argument('--wait-time', type=float, default=2, help='wait time for key press')
    parser.add_argument('--num-samples', type=int, default=1, 
                       help='number of samples to visualize (default: 1)')
    parser.add_argument('--mode', type=str, choices=['test', 'val', 'train'], 
                       default='test', help='running mode (default: test)')
    parser.add_argument(
        '--task',
        type=str,
        default='multi-view_det',
        choices=[
            'mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg',
            'multi-modality_det'
        ],
        help='task type for visualization hook')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='score threshold for visualization')
    return parser.parse_args()


def setup_config(args) -> Config:
    """Setup and return configuration object."""
    cfg = Config.fromfile(args.config)
    
    # Set work directory
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join(
            './work_dirs',
            osp.splitext(osp.basename(args.config))[0])
    
    # Set checkpoint path
    cfg.load_from = args.checkpoint
    
    # Setup visualization if needed
    if args.show:
        cfg = trigger_visualization_hook(cfg, args)
    
    return cfg


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualization_hook['test_out_dir'] = args.show_dir
        all_task_choices = [
            'mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg',
            'multi-modality_det'
        ]
        assert args.task in all_task_choices, 'You must set '\
            f"'--task' in {all_task_choices} in the command " \
            'if you want to use visualization hook'
        visualization_hook['vis_task'] = args.task
        visualization_hook['score_thr'] = args.score_thr
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def visualize_results(model, dataloader, visualizer, num_samples: int = 1):
    """Visualize detection results on test samples.
    
    Args:
        model: The detection model.
        dataloader: The test dataloader.
        visualizer: The visualizer object.
        num_samples: Number of samples to visualize.
    """
    # Set dataset meta for proper class names/palette if available
    if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'metainfo'):
        visualizer.dataset_meta = dataloader.dataset.metainfo
    
    model.eval()
    count = 0
    
    with torch.no_grad():
        for data_batch in dataloader:
            output = model.test_step(data_batch)
            
            # Skip if no predictions
            if not output or (output[0].pred_instances is None):
                continue
            
            # Process each sample in the batch
            for sample in output:
                if sample.pred_instances is None:
                    continue
                
                # Load and visualize image
                if hasattr(sample, 'img_path') and sample.img_path:
                    try:
                        image = np.array(Image.open(sample.img_path).convert('RGB'))
                        visualizer.add_datasample(
                            osp.basename(sample.img_path),
                            image,
                            sample,
                            show=True)
                        count += 1
                        
                        if count >= num_samples:
                            return
                    except Exception as e:
                        print(f"Error processing {sample.img_path}: {e}")
                        continue


def main():
    """Main function for testing."""
    args = parse_args()
    cfg = setup_config(args)
    
    # Create runner
    runner = Runner.from_cfg(cfg)
    
    # Load checkpoint if specified
    if getattr(cfg, 'load_from', None):
        load_checkpoint(runner.model, cfg.load_from, map_location='cpu')
    
    # Run based on mode
    if args.mode == 'test':
        if args.show:
            visualizer = Det3DLocalVisualizer()
            visualize_results(
                runner.model, 
                runner.test_dataloader, 
                visualizer, 
                args.num_samples)
        else:
            runner.test()
    elif args.mode == 'val':
        runner.val()
    elif args.mode == 'train':
        runner.train()


def test_nuscenes():
    from mmdet3d.registry import DATASETS
    args = parse_args()
    cfg = setup_config(args)
    nuscenes_dataset = DATASETS.build(cfg.dataset)


    local_visualizer = MultiFrameDet3DLocalVisualizer()
    local_visualizer.dataset_meta = nuscenes_dataset.metainfo

    data_input = nuscenes_dataset[0]['inputs']
    data_sample = nuscenes_dataset[0]['data_samples']

    breakpoint()

    local_visualizer.add_datasample(name='test_nuscenes',
                                    data_input = data_input,
                                    data_sample = data_sample,
                                    vis_task='multi-modality_det',
                                    draw_gt=True,
                                    draw_pred=False,
                                    show=True,
                                    out_file='work_dirs/test_nuscenes_vis.png',
                                    wait_time=-1)


def test_ckpt():
    args = parse_args()
    cfg = setup_config(args)

    ckpt = torch.load(args.checkpoint, map_location='cpu')['state_dict']

    # remap ckpt
    def remap_attention_keys(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k

            if 'pts_bbox_head' in new_k:
                new_k = new_k.replace('pts_bbox_head.', '')

            if 'transformer' in new_k:
                new_k = new_k.replace('transformer.', '')

            if 'encoder.layers' in new_k:
                if 'encoder' in new_k:
                    new_k = new_k.replace(
                        'encoder', 'bev_encoder'
                    )

                if 'attentions.0' in new_k:
                    # attentions.0 → temporal_self_attn
                    new_k = new_k.replace(
                        "attentions.0.", "temporal_attn."
                    )
                elif 'attentions.1' in new_k:
                    # attentions.1 → spatial_cross_attn
                    new_k = new_k.replace(
                        "attentions.1.", "spatial_cross_attn."
                    )
                elif 'ffns.0' in new_k:
                    new_k = new_k.replace(
                        'ffns.0', 'ffn'
                    )
            elif 'decoder.layers' in new_k:
                if 'attentions.0' in new_k:
                    new_k = new_k.replace(
                        'attentions.0', 'self_attn'
                    )
                elif 'attentions.1' in new_k:
                    new_k = new_k.replace(
                        'attentions.1', 'cross_attn'
                    )
                elif 'ffns.0' in new_k:
                    new_k = new_k.replace(
                        'ffns.0', 'ffn'
                    )
            elif 'cls_branches' in new_k:
                new_k = new_k.replace(
                    'cls_branches', 'bbox_head.cls_branches'
                )
            elif 'reg_branches' in new_k:
                new_k = new_k.replace(
                    'reg_branches', 'bbox_head.reg_branches'
                )
            new_state_dict[new_k] = v

        return new_state_dict

    ckpt = remap_attention_keys(ckpt)

    # # img_backbone
    # img_backbone = MODELS.build(cfg.model.img_backbone)
    # backbone_ckpt = {k.replace('img_backbone.', ''): v for k, v in ckpt.items() if 'img_backbone' in k}
    # missing, unexpected = img_backbone.load_state_dict(backbone_ckpt, strict=False)
    # if len(missing) > 0 or len(unexpected) > 0:
    #     print('[bold red]Some keys did not match for the img_backbone[/bold red]')
    #     print(f'missing keys: {missing}')
    #     print(f'[bold green]unexpected keys: {unexpected}[/bold green]')
    # else:
    #     print('[bold green]All keys matched successfully for the img_backbone[/bold green]')

    # # img_neck
    # img_neck = MODELS.build(cfg.model.img_neck)
    # neck_ckpt = {k.replace('img_neck.', ''): v for k, v in ckpt.items() if 'img_neck' in k}
    # missing, unexpected = img_neck.load_state_dict(neck_ckpt, strict=False)
    # if len(missing) > 0 or len(unexpected) > 0:
    #     print('[bold red]Some keys did not match for the img_neck[/bold red]')
    #     print(f'missing keys: {missing}')
    #     print(f'[bold green]unexpected keys: {unexpected}[/bold green]')
    # else:   
    #     print('[bold green]All keys matched successfully for the img_neck[/bold green]')


    # bevformer_encoder = MODELS.build(cfg.model.encoder)
    # bevformer_encoder_ckpt={k.replace('pts_bbox_head.transformer.encoder.', ''): v
    #                                 for k, v in ckpt.items()
    #                                 if 'pts_bbox_head.transformer.encoder' in k}
    # missing, unexpected = bevformer_encoder.load_state_dict(bevformer_encoder_ckpt, strict=False)
    # if len(missing) > 0 or len(unexpected) > 0:
    #     print('[bold red]Some keys did not match for the bevformer_encoder[/bold red]')
    #     print(f'missing keys: {missing}')
    #     print(f'[bold green]unexpected keys: {unexpected}[/bold green]')
    # else:   
    #     print('[bold green]All keys matched successfully for the bevformer_encoder[/bold green]')

    # bevformer_decoder = MODELS.build(cfg.model.decoder)
    # bevformer_decoder_ckpt={k.replace('pts_bbox_head.transformer.decoder.', ''): v
    #                                 for k, v in ckpt.items()
    #                                 if 'pts_bbox_head.transformer.decoder' in k}
    # missing, unexpected = bevformer_decoder.load_state_dict(bevformer_decoder_ckpt, strict=False)
    # if len(missing) > 0 or len(unexpected) > 0:
    #     print('[bold red]Some keys did not match for the bevformer_decoder[/bold red]')
    #     print(f'missing keys: {missing}')
    #     print(f'[bold green]unexpected keys: {unexpected}[/bold green]')
    # else:   
    #     print('[bold green]All keys matched successfully for the bevformer_decoder[/bold green]')

    bevformer = MODELS.build(cfg.model)
    missing, unexpected = bevformer.load_state_dict(ckpt, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        print('[bold red]Some keys did not match for the bevformer_decoder[/bold red]')
        print(f'missing keys: {missing}')
        print(f'[bold green]unexpected keys: {unexpected}[/bold green]')
    else:   
        print('[bold green]All keys matched successfully for the bevformer_decoder[/bold green]')
    bevformer.cuda()

    nuscenes_dataset = DATASETS.build(cfg.dataset)
    print(f'dataset size: {len(nuscenes_dataset)}')

    local_visualizer = MultiFrameDet3DLocalVisualizer()
    local_visualizer.dataset_meta = nuscenes_dataset.metainfo

    train_dataloader = DataLoader(nuscenes_dataset, batch_size=1, shuffle=False, collate_fn=pseudo_collate)

    bevformer.eval()
    with torch.no_grad():
        for idx, data_batch in enumerate(train_dataloader):
            # print(data_batch.keys())

            inputs = copy.deepcopy(data_batch['inputs'])

            detsamples = bevformer.test_step(data_batch)


            data_input = {
                'img': inputs['img'][0],      # 原来是 [tensor(T, 6, C, H, W)] -> 取第 0 个
                'points': inputs['points'][0] # 原来是 [[points_t-3,...,points_t]] -> 取第 0 个
            }
            data_sample = detsamples[0]


            local_visualizer.add_datasample(name='test_nuscenes',
                                            data_input = data_input,
                                            data_sample = data_sample,
                                            vis_task='multi-modality_det',
                                            draw_gt=False,
                                            draw_pred=True,
                                            show=False,
                                            out_file=f'work_dirs/test_ckpt/test_nuscenes_vis_{idx}.png',
                                            wait_time=-1)


            # break


if __name__ == '__main__':
    # main()
    # test_nuscenes()
    test_ckpt()