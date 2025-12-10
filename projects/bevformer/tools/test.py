import argparse
import os
import os.path as osp

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rich import print

from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint
from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.visualization import DetLocalVisualizer

from mmhdmap.registry import MODELS

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
            visualizer = DetLocalVisualizer()
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


def denormalize_img(img_tensor, img_norm_cfg=None, convert_bgr_to_rgb=False):
    """Denormalize image tensor to [0, 255] uint8.
    
    Args:
        img_tensor (torch.Tensor): Image tensor in [C, H, W] or [N, C, H, W] format.
        img_norm_cfg (dict, optional): Normalization config with 'mean' and 'std'.
        convert_bgr_to_rgb (bool): Whether to convert BGR to RGB. Default: False.
    
    Returns:
        np.ndarray: Denormalized image in [H, W, C] format, uint8.
    """
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.detach().cpu().numpy()
    else:
        img = np.array(img_tensor)
    
    # Handle different tensor shapes
    if img.ndim == 4:  # [N, C, H, W]
        img = img[0]  # Take first image
    if img.ndim == 3 and img.shape[0] == 3:  # [C, H, W]
        img = img.transpose(1, 2, 0)  # [H, W, C]
    
    # Color channel conversion
    # Current frame (frame_idx=0) appears to be in RGB format already
    # Historical frames may need BGR->RGB conversion
    if convert_bgr_to_rgb and img.shape[-1] == 3:
        img = img[..., ::-1]  # Convert BGR to RGB
    
    # Check image value range
    img_min, img_max = img.min(), img.max()
    
    # Denormalize if norm_cfg is provided
    # Note: mmdet3d typically normalizes to [0, 1] range, but some models
    # use ImageNet normalization (mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
    if img_norm_cfg is not None:
        mean = np.array(img_norm_cfg.get('mean', [0, 0, 0]))
        std = np.array(img_norm_cfg.get('std', [1, 1, 1]))
        
        # If mean/std are large (ImageNet style), image is likely already normalized
        # Otherwise, assume image is in [0, 1] range
        if np.abs(mean).max() > 10 or np.abs(std - 1).max() > 0.1:
            # ImageNet normalization: denormalize first
            img = img * std + mean
            # Then normalize to [0, 1]
            img = img / 255.0
        else:
            # Image is likely already in [0, 1] range, just apply denormalization
            img = img * std + mean
    
    # Handle different value ranges
    if img_max > 1.5:
        # Image is likely in [0, 255] range, normalize to [0, 1]
        img = img / 255.0
    elif img_min < 0:
        # Image might be in [-1, 1] range or normalized with negative mean
        # Try to map to [0, 1]
        img = (img - img_min) / (img_max - img_min)
    
    # Clip to [0, 1] and convert to uint8
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    
    return img


def visualize_multi_frame_multi_view(inputs, data_samples, save_path=None, show=True):
    """Visualize multi-frame multi-view images in a grid layout.
    
    Layout: Rows = views (cameras), Columns = frames (temporal)
    
    Args:
        inputs (dict): Input dict containing 'img' key.
            img shape: [T, N_views, C, H, W] or [T, C, H, W]
        data_samples (Det3DDataSample): Data sample containing metainfo.
        save_path (str, optional): Path to save the visualization.
        show (bool): Whether to display the image.
    """
    img_tensor = inputs['img']  # [T, N_views, C, H, W] or [T, C, H, W]
    
    # Get normalization config from metainfo
    img_norm_cfg = None
    if hasattr(data_samples, 'metainfo') and 'img_norm_cfg' in data_samples.metainfo:
        img_norm_cfg = data_samples.metainfo['img_norm_cfg']
    
    # Determine shape
    if img_tensor.ndim == 5:  # [T, N_views, C, H, W]
        num_frames, num_views, C, H, W = img_tensor.shape
        is_multi_view = True
    elif img_tensor.ndim == 4:  # [T, C, H, W] - single view
        num_frames, C, H, W = img_tensor.shape
        num_views = 1
        is_multi_view = False
    else:
        raise ValueError(f"Unexpected img tensor shape: {img_tensor.shape}")
    
    # Camera view names (NuScenes standard order)
    view_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                  'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    # Create grid: rows = views, cols = frames
    fig, axes = plt.subplots(num_views, num_frames, 
                            figsize=(num_frames * 4, num_views * 3),
                            squeeze=False)
    
    # Process each frame and view
    for frame_idx in range(num_frames):
        for view_idx in range(num_views):
            ax = axes[view_idx, frame_idx]
            
            # Extract image for this frame and view
            if is_multi_view:
                frame_view_img = img_tensor[frame_idx, view_idx]  # [C, H, W]
            else:
                frame_view_img = img_tensor[frame_idx]  # [C, H, W]
            
            # Denormalize and convert to numpy
            # Current frame (frame_idx=0) is already in RGB, historical frames need BGR->RGB conversion
            convert_bgr_to_rgb = (frame_idx != 0)
            img_np = denormalize_img(frame_view_img, img_norm_cfg, convert_bgr_to_rgb=convert_bgr_to_rgb)
            
            # Debug: print image stats for first few images
            if frame_idx == 0 and view_idx == 0:
                if isinstance(frame_view_img, torch.Tensor):
                    raw_min, raw_max = frame_view_img.min().item(), frame_view_img.max().item()
                    raw_mean = frame_view_img.mean().item()
                else:
                    raw_min, raw_max = float(frame_view_img.min()), float(frame_view_img.max())
                    raw_mean = float(frame_view_img.mean())
                print(f"  Image stats (Frame {frame_idx}, View {view_idx}):")
                print(f"    Raw tensor: min={raw_min:.4f}, max={raw_max:.4f}, mean={raw_mean:.4f}")
                print(f"    Denormalized: min={img_np.min()}, max={img_np.max()}, mean={img_np.mean():.2f}")
            
            # Display image
            ax.imshow(img_np)
            ax.axis('off')
            
            # Add title: combine view name and frame info
            title_parts = []
            
            # View name (for first column or all if single frame)
            if view_idx < len(view_names):
                view_name = view_names[view_idx]
            else:
                view_name = f'View {view_idx}'
            
            # Frame info
            if frame_idx == 0:
                frame_info = 'Current (t=0)'
            else:
                frame_info = f'Frame {frame_idx}'
            
            # Combine: show both view and frame for clarity
            title = f'{view_name}\n{frame_info}'
            ax.set_title(title, fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        save_dir = osp.dirname(save_path)
        if save_dir and not osp.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def test_nuscenes():
    from mmdet3d.registry import MODELS
    args = parse_args()
    cfg = setup_config(args)
    nuscenes_dataset = MODELS.build(cfg.dataset)
    
    print(f"Dataset length: {len(nuscenes_dataset)}")
    
    # Get sample data
    sample_idx = 10
    sample = nuscenes_dataset[sample_idx]
    inputs = sample['inputs']
    data_samples = sample['data_samples']
    
    # Print data structure info
    print(f"\nSample {sample_idx} structure:")
    print(f"  inputs keys: {list(inputs.keys())}")
    if 'img' in inputs:
        print(f"  img shape: {inputs['img'].shape}")
        print(f"  img dtype: {inputs['img'].dtype}")
    
    if hasattr(data_samples, 'metainfo'):
        print(f"  metainfo keys: {list(data_samples.metainfo.keys())}")
        if 'img_norm_cfg' in data_samples.metainfo:
            print(f"  img_norm_cfg: {data_samples.metainfo['img_norm_cfg']}")
    
    if hasattr(data_samples, 'gt_instances_3d'):
        gt = data_samples.gt_instances_3d
        print(f"  gt_instances_3d attributes: {[attr for attr in dir(gt) if not attr.startswith('_')]}")
        if hasattr(gt, 'bboxes_3d'):
            print(f"  gt_bboxes_3d: {gt.bboxes_3d}")
        if hasattr(gt, 'labels_3d'):
            print(f"  gt_labels_3d: {gt.labels_3d}")
    
    # Visualize
    print("\nVisualizing multi-frame multi-view images...")
    save_path = osp.join(cfg.work_dir, f'sample_{sample_idx}_visualization.png')
    visualize_multi_frame_multi_view(
        inputs, 
        data_samples, 
        save_path=save_path,
        show=True
    )

def test_ckpt():
    args = parse_args()
    cfg = setup_config(args)

    ckpt = torch.load(args.checkpoint, map_location='cpu')['state_dict']

    # img_backbone
    img_backbone = MODELS.build(cfg.model.img_backbone)
    backbone_ckpt = {k.replace('img_backbone.', ''): v for k, v in ckpt.items() if 'img_backbone' in k}
    missing, unexpected = img_backbone.load_state_dict(backbone_ckpt, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        print('[bold red]Some keys did not match for the img_backbone[/bold red]')
        print(f'missing keys: {missing}')
        print(f'[bold green]unexpected keys: {unexpected}[/bold green]')
    else:
        print('[bold green]All keys matched successfully for the img_backbone[/bold green]')

    # img_neck
    img_neck = MODELS.build(cfg.model.img_neck)
    neck_ckpt = {k.replace('img_neck.', ''): v for k, v in ckpt.items() if 'img_neck' in k}
    missing, unexpected = img_neck.load_state_dict(neck_ckpt, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        print('[bold red]Some keys did not match for the img_neck[/bold red]')
        print(f'missing keys: {missing}')
        print(f'[bold green]unexpected keys: {unexpected}[/bold green]')
    else:   
        print('[bold green]All keys matched successfully for the img_neck[/bold green]')

    temporal_self_attn = MODELS.build(cfg.model.temporal_self_attn)
    print(f'temporal_self_attn: {temporal_self_attn}')
    ms_attn_3d = MODELS.build(cfg.model.ms_attn_3d)
    print(f'ms_attn_3d: {ms_attn_3d}')

if __name__ == '__main__':
    # main()
    # test_nuscenes()
    test_ckpt()