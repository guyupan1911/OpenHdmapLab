from typing import List, Optional, Union
import numpy as np

from mmcv.transforms.base import BaseTransform

from mmhdmap.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PrintDict(BaseTransform):

    def transform(self, results: dict) -> Optional[dict]:
        print("=" * 80)
        print("PrintDict Debug Info:")
        print("=" * 80)
        
        print(f'ann_info keys: {results["ann_info"].keys()}')

        if 'multi_frame_data' in results:
            multi_frame_data = results['multi_frame_data']
            print(f"multi_frame_data type: {type(multi_frame_data)}")
            print(f"multi_frame_data keys (frame indices): {list(multi_frame_data.keys())}")
            
            for frame_idx, frame_data in multi_frame_data.items():
                print(f"\n--- Frame {frame_idx} ---")
                print(f"  frame_data keys: {list(frame_data.keys())}")
                
                if 'img' in frame_data:
                    img = frame_data['img']
                    print(f"  img type: {type(img)}")
                    if isinstance(img, dict):
                        print(f"  img (dict) keys: {list(img.keys())}")
                        for cam_name, cam_img in img.items():
                            if isinstance(cam_img, np.ndarray):
                                print(f"    {cam_name}: numpy array, shape={cam_img.shape}, dtype={cam_img.dtype}")
                            else:
                                print(f"    {cam_name}: {type(cam_img)}")
                    elif isinstance(img, list):
                        print(f"  img (list) length: {len(img)}")
                        for i, cam_img in enumerate(img):
                            if isinstance(cam_img, np.ndarray):
                                print(f"    [{i}]: numpy array, shape={cam_img.shape}, dtype={cam_img.dtype}")
                            else:
                                print(f"    [{i}]: {type(cam_img)}")
                    elif isinstance(img, np.ndarray):
                        print(f"  img: numpy array, shape={img.shape}, dtype={img.dtype}")
                    else:
                        print(f"  img: {type(img)}")
                
                # Print other important fields
                for key in ['cam2img', 'lidar2cam', 'lidar2img', 'img_shape', 'ori_shape']:
                    if key in frame_data:
                        val = frame_data[key]
                        if isinstance(val, (list, tuple)) and len(val) > 0:
                            print(f"  {key}: {type(val)}, length={len(val)}, first element type={type(val[0])}")
                        elif isinstance(val, np.ndarray):
                            print(f"  {key}: numpy array, shape={val.shape}, dtype={val.dtype}")
                        else:
                            print(f"  {key}: {type(val)}")
        else:
            print("No multi_frame_data found")
            print(f"results keys: {list(results.keys())}")
            
            if 'img' in results:
                img = results['img']
                print(f"img type: {type(img)}")
                if isinstance(img, dict):
                    print(f"img (dict) keys: {list(img.keys())}")
                elif isinstance(img, list):
                    print(f"img (list) length: {len(img)}")
                elif isinstance(img, np.ndarray):
                    print(f"img: numpy array, shape={img.shape}, dtype={img.dtype}")
        
        print("=" * 80)
        return results