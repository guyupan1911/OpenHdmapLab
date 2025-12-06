from collections import OrderedDict
from typing import List, Sequence, Union

import numpy as np
from numpy import dtype
import torch
import mmengine
from mmcv.transforms.base import BaseTransform
from mmengine.structures import InstanceData
from mmdet3d.structures import Det3DDataSample

from mmhdmap.registry import TRANSFORMS


def to_tensor(
    data: Union[torch.Tensor, np.ndarray, Sequence, int, float]) -> torch.Tensor:
    """Convert objects of various python types to :obj:`torch.Tensor`.
    
    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    
    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    
    Returns:
        torch.Tensor: the converted data.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        if data.dtype is dtype('float64'):
            data = data.astype(np.float32)
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmengine.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@TRANSFORMS.register_module()
class PackMultiFrame3DDetInputs(BaseTransform):
    """
    Pack multi-frame 3D detection inputs.

    This transform collects all frames (current + historical) images,
    stacks them, and packs them together with data_samples for model input.

    Args:
        keys (list[str]): The keys to be packed. Default: ['img', 'gt_bboxes_3d', 'gt_labels_3d'].
        meta_keys (list[str]): Keys to save into data_samples.metainfo for each frame.
    """
    def __init__(self,
                 keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'],
                 meta_keys=['lidar2img', 'lidar2cam', 'cam2img', 'img_shape']):
        self.keys = keys
        self.meta_keys = meta_keys

    def transform(self, results: Union[dict, List[dict]]) -> Union[dict, List[dict]]:
        """Method to pack the input data. When the value in this dict is a
        list, it usually is in Augmentations Testing.
        
        Args:
            results (dict | list[dict]): Result dict from the data pipeline.
        
        Returns:
            dict | list[dict]:
            - 'inputs' (dict): The forward data of models. It usually contains
              following keys:
                - img
            - 'data_samples' (:obj:`Det3DDataSample`): The annotation info of
              the sample.
        """
        # augtest: handle list of results
        if isinstance(results, list):
            if len(results) == 1:
                # simple test
                return self.pack_single_results(results[0])
            pack_results = []
            for single_result in results:
                pack_results.append(self.pack_single_results(single_result))
            return pack_results
        
        # normal training and simple testing
        elif isinstance(results, dict):
            return self.pack_single_results(results)
        else:
            raise NotImplementedError(f'Unsupported type {type(results)}')
    
    def pack_single_results(self, results: dict) -> dict:
        """Method to pack the single input data.
        
        Args:
            results (dict): Result dict from the data pipeline.
        
        Returns:
            dict: A dict contains
            - 'inputs' (dict): The forward data of models. It usually contains
              following keys:
                - img
            - 'data_samples' (:obj:`Det3DDataSample`): The annotation info
              of the sample.
        """
        # Single-frame case (no multi_frame_data)
        if 'multi_frame_data' not in results:
            data = {key: results[key] for key in self.keys if key in results}
            # Save metainfo
            if 'data_samples' in results:
                ds = results['data_samples']
                if not hasattr(ds, 'metainfo'):
                    ds.metainfo = {}
                for key in self.meta_keys:
                    if key in results:
                        ds.metainfo[key] = results[key]
                data['data_samples'] = ds
            return data

        # Multi-frame case
        multi_frame_data = results['multi_frame_data']
        if not isinstance(multi_frame_data, OrderedDict):
            multi_frame_data = OrderedDict(sorted(multi_frame_data.items()))

        imgs_list = []
        # Store metadata for each frame (avoid deepcopy for performance)
        frame_metas = []

        for frame_idx, frame_data in multi_frame_data.items():
            img = frame_data.get('img', None)
            if img is None:
                raise ValueError(f'Frame {frame_idx} does not have "img" key.')
            
            # Convert numpy arrays to tensors and handle format conversion
            # Reference: mmdet3d Pack3DDetInputs
            if isinstance(img, list):
                # Multi-view case: process multiple imgs in single frame
                imgs = np.stack(img, axis=0)  # [N_views, H, W, C]
                # To improve computational speed, use torch.permute() rather than np.transpose()
                if imgs.flags.c_contiguous:
                    imgs = to_tensor(imgs).permute(0, 3, 1, 2).contiguous()  # [N_views, C, H, W]
                else:
                    imgs = to_tensor(np.ascontiguousarray(imgs.transpose(0, 3, 1, 2)))
                img = imgs
            elif isinstance(img, np.ndarray):
                # Single image case
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                # Use torch.permute() for better performance
                if img.flags.c_contiguous:
                    img = to_tensor(img).permute(2, 0, 1).contiguous()  # [C, H, W]
                else:
                    img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            elif isinstance(img, torch.Tensor):
                # Already a tensor, ensure correct format
                if img.dtype != torch.float32:
                    img = img.float()
                # Ensure CHW format if needed
                if img.ndim == 3 and img.shape[0] != 3:
                    img = img.permute(2, 0, 1).contiguous()
            else:
                img = to_tensor(img)
            
            imgs_list.append(img)

            # Collect per-frame metadata (avoid deepcopy for performance)
            frame_meta = {}
            for key in self.meta_keys:
                if key in frame_data:
                    # Direct reference for numpy arrays and basic types (no deepcopy)
                    frame_meta[key] = frame_data[key]
            frame_meta['frame_idx'] = frame_idx
            frame_metas.append(frame_meta)

        # Stack all frames -> [N_frames, N_views, C, H, W]
        # Note: BEVFormer expects [B, N_view, T, C, H, W] after batching
        stacked_imgs = torch.stack(imgs_list, dim=0)  # [T, N_views, C, H, W]
        
        # Get ann_info from results (same level as multi_frame_data)
        ann_info = results.get('ann_info', None)
        if ann_info is None:
            raise ValueError('ann_info is required in results for multi-frame data packing.')
        
        # Create Det3DDataSample
        data_samples = Det3DDataSample()
        
        # Create InstanceData for gt_instances_3d
        gt_instances_3d = InstanceData()
        
        # Set ground truth from ann_info (convert to tensor if needed)
        # Note: gt_bboxes_3d might already be a BaseInstance3DBoxes object (e.g., LiDARInstance3DBoxes)
        # In that case, we should assign it directly without conversion
        if 'gt_bboxes_3d' in ann_info:
            bboxes_3d = ann_info['gt_bboxes_3d']
            # Check if it's already a BaseInstance3DBoxes object
            from mmdet3d.structures import BaseInstance3DBoxes
            if isinstance(bboxes_3d, BaseInstance3DBoxes):
                gt_instances_3d.bboxes_3d = bboxes_3d
            else:
                # Convert numpy array or other types to tensor
                gt_instances_3d.bboxes_3d = to_tensor(bboxes_3d)
        if 'gt_labels_3d' in ann_info:
            gt_instances_3d.labels_3d = to_tensor(ann_info['gt_labels_3d'])
        if 'velocities' in ann_info:
            gt_instances_3d.velocities = to_tensor(ann_info['velocities'])
        if 'bbox_3d_isvalid' in ann_info:
            gt_instances_3d.bbox_3d_isvalid = to_tensor(ann_info['bbox_3d_isvalid'])
        
        # Assign to data_samples
        data_samples.gt_instances_3d = gt_instances_3d
        
        # Get current frame (frame_idx=0) for reference
        current_frame = multi_frame_data.get(0, multi_frame_data[list(multi_frame_data.keys())[-1]])
        
        # Prepare metainfo dict for current frame
        metainfo_dict = {}
        
        # Save other annotation info to metainfo
        for key in ['num_lidar_pts', 'num_radar_pts', 'bbox_3d_isvalid', 'instances']:
            if key in ann_info:
                metainfo_dict[key] = ann_info[key]
        
        # Save img_norm_cfg if available (for normalization)
        if 'img_norm_cfg' in current_frame:
            metainfo_dict['img_norm_cfg'] = current_frame['img_norm_cfg']
        
        # Save current frame (frame_idx=0) metadata to metainfo
        if 0 in multi_frame_data:
            current_meta = multi_frame_data[0]
            for key in self.meta_keys:
                if key in current_meta:
                    metainfo_dict[key] = current_meta[key]
            
            # Save additional metadata from current frame if available
            for key in ['timestamp', 'ego2global_translation', 'ego2global_rotation',
                       'lidar2ego_translation', 'lidar2ego_rotation']:
                if key in current_meta:
                    metainfo_dict[key] = current_meta[key]
        
        # Use set_metainfo() method (required for Det3DDataSample)
        data_samples.set_metainfo(metainfo_dict)
        
        # Save historical frames metadata as structured list
        # Each element is a dict containing that frame's metadata
        # This format is compatible with BEVFormer/StreamPETR and easy to extend
        history_metas = []
        for frame_idx in sorted(multi_frame_data.keys()):
            if frame_idx != 0:  # Skip current frame (frame_idx=0)
                hist_frame_data = multi_frame_data[frame_idx]
                hist_meta = {}
                
                # Collect metadata keys
                for key in self.meta_keys:
                    if key in hist_frame_data:
                        hist_meta[key] = hist_frame_data[key]
                
                # Add frame index
                hist_meta['frame_idx'] = frame_idx
                
                # Add additional metadata if available
                for key in ['timestamp', 'ego2global_translation', 'ego2global_rotation',
                           'lidar2ego_translation', 'lidar2ego_rotation', 'img_shape', 
                           'ori_shape', 'pad_shape', 'scale_factor']:
                    if key in hist_frame_data:
                        hist_meta[key] = hist_frame_data[key]
                
                history_metas.append(hist_meta)
        
        # Store history metadata as a field in data_samples
        # This allows easy access and extension for velocity, can_bus, etc.
        if history_metas:
            data_samples.set_field(history_metas, 'history_metas', field_type='metainfo')

        return {
            'inputs': {'img': stacked_imgs},
            'data_samples': data_samples
        }
