import copy
from typing import Union, List, Optional, Tuple
from collections import OrderedDict

from mmcv.transforms.base import BaseTransform
from mmengine.dataset import Compose

from mmhdmap.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadMultiFrameData(BaseTransform):

    def __init__(self, transforms: List[dict]) -> None:
        self.transforms = Compose(transforms)
    
    def transform(self, results: dict) -> dict:
        
        assert 'multi_frame_data' in results, "multi_frame_data not found in results"
        
        multi_frame_data = results['multi_frame_data']
        # Ensure multi_frame_data is OrderedDict and sorted by frame_idx
        # This ensures frames are processed in chronological order (negative -> 0)
        if not isinstance(multi_frame_data, OrderedDict):
            multi_frame_data = OrderedDict(sorted(multi_frame_data.items()))
        else:
            # If already OrderedDict, ensure it's sorted by frame_idx
            # Sort to ensure chronological order (e.g., -3, -2, -1, 0)
            multi_frame_data = OrderedDict(sorted(multi_frame_data.items()))

        processed_frames = OrderedDict()
        for frame_idx, frame_input in multi_frame_data.items():
            frame_result = self.transforms(copy.deepcopy(frame_input))
            processed_frames[frame_idx] = frame_result
        
        results['multi_frame_data'] = processed_frames
        return results