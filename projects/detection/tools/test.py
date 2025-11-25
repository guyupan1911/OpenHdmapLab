import argparse
import os.path as osp

import torch
import numpy as np
from PIL import Image

from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint
from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.visualization import DetLocalVisualizer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='MMDetection test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
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


if __name__ == '__main__':
    main()