from typing import List, Optional, Union
import numpy as np

from mmcv.transforms.base import BaseTransform

from mmhdmap.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PrintDict(BaseTransform):

    def transform(self, results: dict) -> Optional[dict]:
        print(results.keys())
        print(results['inputs'].keys())
        print(results['data_samples'].keys())
        print(results['data_samples'].metainfo.keys())
        # print(f"img type :{type(results['img'])}")
        # print(results['img'].shape)
      
        return results