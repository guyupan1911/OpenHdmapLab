from typing import List, Optional, Union

from mmcv.transforms import BaseTransform

from mmhdmap.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PrintDict(BaseTransform):

    def transform(self, results: dict) -> Optional[dict]:
        print(len(results['multi_frame_data']))
  
        return results