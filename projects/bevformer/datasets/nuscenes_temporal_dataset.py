import copy
from collections import OrderedDict
from typing import Union, List, Optional, Tuple

import torch
import numpy as np
from nuscenes.eval.common.utils import Quaternion, quaternion_yaw

from mmdet3d.datasets import NuScenesDataset
from mmhdmap.registry import DATASETS, TRANSFORMS


@DATASETS.register_module()
class NuScenesTemporalDataset(NuScenesDataset):

    def __init__(self,
                 frames: Tuple[int, ...] = (),
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.frames = frames
    
    def process_canbus(self, data_info):
        """Populate can_bus fields to match original BEVFormer conventions.

        Original BEVFormer (mmdet3d plugin) sets:
        - can_bus[:3]    = ego2global translation (x, y, z)
        - can_bus[3:7]   = ego2global rotation quaternion (w, x, y, z)
        - can_bus[-2]    = ego yaw in radians (wrapped to [0, 2pi))
        - can_bus[-1]    = ego yaw in degrees (wrapped to [0, 360))
        """
        # Prefer quaternion+translation fields if they exist (more stable than
        # reconstructing from a float matrix that may not be perfectly orthogonal).
        if 'ego2global_rotation' in data_info and 'ego2global_translation' in data_info:
            rotation = Quaternion(data_info['ego2global_rotation'])
            translation = np.asarray(data_info['ego2global_translation'], dtype=np.float64)
        else:
            ego2global = np.asarray(data_info['ego2global'], dtype=np.float64)
            translation = ego2global[:3, 3]
            rot_mat = ego2global[:3, :3]
            # Orthonormalize rot_mat to avoid pyquaternion's strict orthogonality check.
            # Use SVD-based projection onto SO(3).
            U, _, Vt = np.linalg.svd(rot_mat)
            rot_mat = U @ Vt
            if np.linalg.det(rot_mat) < 0:
                U[:, -1] *= -1
                rot_mat = U @ Vt
            rotation = Quaternion(matrix=rot_mat)
        patch_angle = quaternion_yaw(rotation) / np.pi * 180.0
        if patch_angle < 0:
            patch_angle += 360.0

        can_bus = data_info['can_bus'].copy()
        can_bus[:3] = translation
        # Store quaternion as (w, x, y, z) to match original BEVFormer.
        can_bus[3:7] = rotation.elements
        can_bus[-2] = patch_angle / 180.0 * np.pi  # rad in [0, 2pi)
        can_bus[-1] = patch_angle                  # deg in [0, 360)

        data_info['can_bus'] = can_bus
        return data_info

    def prepare_data(self, index: int) -> Union[dict, None]:
        # 1. Get current frame data_info
        data_info = self.get_data_info(index)
        if data_info is None:
            return None
        cur_scene_token = data_info['scene_token']

        # 2. Check empty annotations
        if not self.test_mode and self.filter_empty_gt:
            if len(data_info['ann_info']['gt_labels_3d']) == 0:
                return None
        
        # 3. Build multi_frame_inputs: Ordereddict with frame_idx as keys
        multi_frame_inputs = OrderedDict()

        # Current frame (frame_idx=0)
        current_input = data_info
        current_input['box_type_3d'] = self.box_type_3d
        current_input['box_mode_3d'] = self.box_mode_3d
        # Add Annotation info for current frame
        if not self.test_mode:
            current_input['ann_info'] = data_info['ann_info']
        
        current_input = self.process_canbus(current_input)

        multi_frame_inputs[0] = current_input

        # Historical frames
        for frame_idx in self.frames:
            chosen_idx = index + frame_idx

            if frame_idx == 0 or chosen_idx < 0 or chosen_idx >= self.__len__():
                continue

            # Get historical frame info
            info = self.get_data_info(chosen_idx)
            # Only load frames from the same scene
            if info['scene_token'] != cur_scene_token:
                continue
        
            # Build input dict for historical frame
            hist_input = info
            hist_input['box_type_3d'] = self.box_type_3d
            hist_input['box_mode_3d'] = self.box_mode_3d
            # Share annotation info for pipeline consistency, not used
            # if not self.test_mode:
            #     hist_input['ann_info'] = copy.deepcopy(data_info['ann_info'])
            hist_input = self.process_canbus(hist_input)
            multi_frame_inputs[frame_idx] = hist_input
        
        result = {
            'multi_frame_data': multi_frame_inputs,
            'ann_info': data_info['ann_info']
        }

        return self.pipeline(result)



