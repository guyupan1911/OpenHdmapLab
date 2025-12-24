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
        """
        Remap checkpoint keys from old BEVFormer structure to new structure.
        
        Old structure: pts_bbox_head.transformer.{encoder, decoder, ...}
        New structure: {bev_encoder, bbox_head.decoder, ...}
        
        Mapping rules:
        1. Remove 'pts_bbox_head.' prefix
        2. Map transformer.encoder.* → bev_encoder.*
           - encoder.layers.*.attentions.0.* → layers.*.temporal_attn.*
           - encoder.layers.*.attentions.1.* → layers.*.spatial_cross_attn.*
           - encoder.layers.*.ffns.0.* → layers.*.ffn.*
        3. Map transformer.decoder.* → bbox_head.decoder.*
           - decoder.layers.*.attentions.0.attn.* → decoder.layers.*.self_attn.attn.*
           - decoder.layers.*.attentions.1.* → decoder.layers.*.cross_attn.*
           - decoder.layers.*.ffns.0.* → decoder.layers.*.ffn.*
        4. Map transformer components to bev_encoder:
           - transformer.level_embeds → bev_encoder.level_embeds
           - transformer.cams_embeds → bev_encoder.cams_embeds
           - transformer.can_bus_mlp.* → bev_encoder.can_bus_mlp.*
        5. Map transformer.reference_points.* → bbox_head.reference_points.*
        6. Map positional_encoding.* → bev_encoder.positional_encoding.*
        7. Map bev_embedding.* → bev_encoder.bev_embedding.*
        8. Map other bbox_head components
        """
        new_state_dict = {}
        
        for k, v in state_dict.items():
            new_k = k
            
            # Step 1: Remove 'pts_bbox_head.' prefix
            if new_k.startswith('pts_bbox_head.'):
                new_k = new_k[len('pts_bbox_head.'):]
            
            # Step 2: Handle transformer.encoder → bev_encoder
            if new_k.startswith('transformer.encoder.'):
                # Remove 'transformer.' prefix and replace 'encoder' with 'bev_encoder'
                new_k = new_k.replace('transformer.encoder.', 'bev_encoder.', 1)
                
                # Map attention and ffn layers within encoder layers
                if '.layers.' in new_k:
                    # Map attentions.0 → temporal_attn
                    if '.attentions.0.' in new_k:
                        new_k = new_k.replace('.attentions.0.', '.temporal_attn.', 1)
                    elif new_k.endswith('.attentions.0'):
                        new_k = new_k[:-len('.attentions.0')] + '.temporal_attn'
                    # Map attentions.1 → spatial_cross_attn (including deformable_attention)
                    elif '.attentions.1.' in new_k:
                        new_k = new_k.replace('.attentions.1.', '.spatial_cross_attn.', 1)
                    elif new_k.endswith('.attentions.1'):
                        new_k = new_k[:-len('.attentions.1')] + '.spatial_cross_attn'
                    # Map ffns.0 → ffn
                    elif '.ffns.0.' in new_k:
                        new_k = new_k.replace('.ffns.0.', '.ffn.', 1)
                    elif new_k.endswith('.ffns.0'):
                        new_k = new_k[:-len('.ffns.0')] + '.ffn'
            
            # Step 3: Handle transformer.decoder → bbox_head.decoder
            elif new_k.startswith('transformer.decoder.'):
                # Remove 'transformer.' prefix and add 'bbox_head.' prefix
                new_k = new_k.replace('transformer.decoder.', 'bbox_head.decoder.', 1)
                
                # Map attention and ffn layers within decoder.layers
                if 'decoder.layers.' in new_k:
                    # Map attentions.0.attn.* → self_attn.attn.*
                    if '.attentions.0.attn.' in new_k:
                        new_k = new_k.replace('.attentions.0.attn.', '.self_attn.attn.', 1)
                    elif new_k.endswith('.attentions.0.attn'):
                        new_k = new_k[:-len('.attentions.0.attn')] + '.self_attn.attn'
                    # Map attentions.0.* (other keys) → self_attn.*
                    elif '.attentions.0.' in new_k:
                        new_k = new_k.replace('.attentions.0.', '.self_attn.', 1)
                    elif new_k.endswith('.attentions.0'):
                        new_k = new_k[:-len('.attentions.0')] + '.self_attn'
                    # Map attentions.1 → cross_attn
                    elif '.attentions.1.' in new_k:
                        new_k = new_k.replace('.attentions.1.', '.cross_attn.', 1)
                    elif new_k.endswith('.attentions.1'):
                        new_k = new_k[:-len('.attentions.1')] + '.cross_attn'
                    # Map ffns.0 → ffn
                    elif '.ffns.0.' in new_k:
                        new_k = new_k.replace('.ffns.0.', '.ffn.', 1)
                    elif new_k.endswith('.ffns.0'):
                        new_k = new_k[:-len('.ffns.0')] + '.ffn'
            
            # Step 4: Handle transformer components → bev_encoder
            elif new_k.startswith('transformer.level_embeds'):
                new_k = new_k.replace('transformer.level_embeds', 'bev_encoder.level_embeds', 1)
            elif new_k.startswith('transformer.cams_embeds'):
                new_k = new_k.replace('transformer.cams_embeds', 'bev_encoder.cams_embeds', 1)
            elif new_k.startswith('transformer.can_bus_mlp.'):
                new_k = new_k.replace('transformer.can_bus_mlp.', 'bev_encoder.can_bus_mlp.', 1)
            
            # Step 5: Handle transformer.reference_points → bbox_head.reference_points
            elif new_k.startswith('transformer.reference_points.'):
                new_k = new_k.replace('transformer.reference_points.', 'bbox_head.reference_points.', 1)
            elif new_k == 'transformer.reference_points':
                new_k = 'bbox_head.reference_points'
            
            # Step 6: Handle positional_encoding → bev_encoder.positional_encoding
            elif new_k.startswith('positional_encoding.'):
                new_k = 'bev_encoder.' + new_k
            elif new_k == 'positional_encoding':
                new_k = 'bev_encoder.positional_encoding'
            
            # Step 7: Handle bev_embedding → bev_encoder.bev_embedding
            elif new_k.startswith('bev_embedding.'):
                new_k = 'bev_encoder.' + new_k
            elif new_k == 'bev_embedding':
                new_k = 'bev_encoder.bev_embedding'
            
            # Step 8: Handle other bbox_head components
            elif 'cls_branches' in new_k:
                if not new_k.startswith('bbox_head.'):
                    new_k = 'bbox_head.' + new_k
            
            elif 'reg_branches' in new_k:
                if not new_k.startswith('bbox_head.'):
                    new_k = 'bbox_head.' + new_k
            
            elif 'code_weights' in new_k:
                if not new_k.startswith('bbox_head.'):
                    new_k = 'bbox_head.' + new_k
            
            elif 'query_embedding' in new_k:
                if not new_k.startswith('bbox_head.'):
                    new_k = 'bbox_head.' + new_k
            
            # Keep img_backbone and img_neck as is (they don't need remapping)
            # All other keys that don't match above patterns are kept as is
          
            new_state_dict[new_k] = v
        
        return new_state_dict

    ckpt = remap_attention_keys(ckpt)

    bevformer = MODELS.build(cfg.model)

    missing, unexpected = bevformer.load_state_dict(ckpt, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        print('[bold red]Some keys did not match for the bevformer[/bold red]')
        print(f'missing keys: {missing}')
        print(f'[bold green]unexpected keys: {unexpected}[/bold green]')
    else:   
        print('[bold green]All keys matched successfully for the bevformer[/bold green]')
    bevformer.cuda()

    nuscenes_dataset = DATASETS.build(cfg.dataset)
    print(f'dataset size: {len(nuscenes_dataset)}')

    local_visualizer = MultiFrameDet3DLocalVisualizer()
    local_visualizer.dataset_meta = nuscenes_dataset.metainfo

    train_dataloader = DataLoader(nuscenes_dataset, batch_size=1, shuffle=False, collate_fn=pseudo_collate)

    bevformer.eval()
    with torch.no_grad():
        for idx, data_batch in enumerate(train_dataloader):
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
                                            show=True,
                                            out_file=f'work_dirs/test_ckpt/test_nuscenes_vis_{idx}.png',
                                            wait_time=-1)

            break
            


if __name__ == '__main__':
    # main()
    # test_nuscenes()
    test_ckpt()