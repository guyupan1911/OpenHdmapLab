import copy
from collections import OrderedDict
from typing import List

from mmcv.transforms.base import BaseTransform
from mmengine.dataset import Compose
from mmdet3d.datasets.transforms.transforms_3d import MultiViewWrapper

from mmhdmap.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MultiFrameWrapper(BaseTransform):

    def __init__(
        self,
        transforms: List[dict],
        override_aug_config: bool = True,
        process_fields: list = ['img', 'cam2img', 'lidar2cam', 'lidar2img'],
        collected_keys: list = [
            'scale', 'scale_factor', 'crop', 'img_crop_offset', 'ori_shape',
            'pad_shape', 'img_shape', 'pad_fixed_size', 'pad_size_divisor',
            'flip', 'flip_direction', 'rotate', 'aug_param'
        ],
        randomness_keys: list = [
            'scale', 'scale_factor', 'crop_size', 'img_crop_offset', 'flip',
            'flip_direction', 'photometric_param', 'aug_param'
        ]
    ) -> None:
        # Create MultiViewWrapper to handle multi-view processing for each frame
        self.multi_view_wrapper = MultiViewWrapper(
            transforms=transforms,
            override_aug_config=override_aug_config,
            process_fields=process_fields,
            collected_keys=collected_keys,
            randomness_keys=randomness_keys
        )
        self.override_aug_config = override_aug_config
        self.collected_keys = collected_keys
        self.process_fields = process_fields
        self.randomness_keys = randomness_keys
    
    def transform(self, results: dict) -> dict:

        if 'multi_frame_data' not in results:
            return self.transforms(results)
        
        multi_frame_data = results['multi_frame_data']
        if not isinstance(multi_frame_data, OrderedDict):
            multi_frame_data = OrderedDict(sorted(multi_frame_data.items()))
        
        for key in self.collected_keys:
            if key not in results:
                results[key] = {}
        
        prev_process_dict = {}
        sorted_frame_indices = sorted(multi_frame_data.keys())
        first_frame_idx = sorted_frame_indices[0] if sorted_frame_indices else None

        for frame_idx in sorted_frame_indices:
            frame_data = multi_frame_data[frame_idx]
            # Copy all fields from frame_data to ensure transforms have access to all metadata
            process_dict = copy.deepcopy(frame_data)

            # Override randomness from previous frame if needed
            # For multi-frame consistency, we need to ensure all frames use the same randomness
            if frame_idx != first_frame_idx and self.override_aug_config:
                # Inject randomness keys from previous frame into process_dict
                # This ensures the first view of current frame uses previous frame's randomness
                for key in self.randomness_keys:
                    if key in prev_process_dict:
                        process_dict[key] = copy.deepcopy(prev_process_dict[key])
            
            # Use MultiViewWrapper to handle multi-view processing for this frame
            # MultiViewWrapper will:
            # 1. Process first view (may generate new randomness if not in process_dict)
            # 2. Process other views (using randomness from first view via override_aug_config)
            process_dict = self.multi_view_wrapper.transform(process_dict)
            
            # For multi-frame consistency: override randomness in process_dict with prev_frame's
            # This ensures that even if transform generated new randomness, we use the consistent one
            if frame_idx != first_frame_idx and self.override_aug_config:
                for key in self.randomness_keys:
                    if key in prev_process_dict:
                        # Override the randomness in process_dict to ensure consistency
                        if isinstance(process_dict.get(key), list):
                            # For list fields, set all elements to the same value
                            prev_val = copy.deepcopy(prev_process_dict[key])
                            process_dict[key] = [prev_val] * len(process_dict[key])
                        else:
                            process_dict[key] = copy.deepcopy(prev_process_dict[key])
            
            # Extract randomness from the first view for next frame
            # For first frame, this will be the newly generated randomness
            # For subsequent frames, this will be the consistent randomness we enforced
            prev_process_dict = {}
            for key in self.randomness_keys:
                if key in process_dict:
                    # For list fields, get the first element (first view)
                    if isinstance(process_dict[key], list) and len(process_dict[key]) > 0:
                        prev_process_dict[key] = copy.deepcopy(process_dict[key][0])
                    elif not isinstance(process_dict[key], list):
                        prev_process_dict[key] = copy.deepcopy(process_dict[key])
            
            # Update frame_data with processed data
            for key in self.process_fields:
                if key in process_dict:
                    frame_data[key] = process_dict[key]
            
            # Collect augmentation-related keys
            for key in self.collected_keys:
                if key in process_dict:
                    results[key][frame_idx] = process_dict[key]
        
        # Store augmentation parameters in results for later use
        if 'aug_param' in prev_process_dict:
            results['aug_param'] = prev_process_dict['aug_param']
        
        return results        