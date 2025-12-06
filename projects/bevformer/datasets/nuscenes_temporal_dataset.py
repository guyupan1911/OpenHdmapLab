import copy
from collections import OrderedDict
from typing import Union, List, Optional, Tuple

import torch
import numpy as np
from nuscenes.eval.common.utils import Quaternion

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
            multi_frame_inputs[frame_idx] = hist_input
        
        result = {
            'multi_frame_data': multi_frame_inputs,
            'ann_info': data_info['ann_info']
        }

        return self.pipeline(result)



