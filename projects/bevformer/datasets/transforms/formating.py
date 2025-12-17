from collections import OrderedDict
from typing import List, Sequence, Union

import numpy as np
import torch

from mmdet3d.datasets.transforms import Pack3DDetInputs

from mmhdmap.registry import TRANSFORMS



@TRANSFORMS.register_module()
class PackMultiFrame3DDetInputs(Pack3DDetInputs):
    """
    Pack multi-frame 3D detection inputs.

    This transform collects all frames (current + historical) images,
    stacks them, and packs them together with data_samples for model input.

    Args:
        keys (list[str]): The keys to be packed. Default: ['img', 'gt_bboxes_3d', 'gt_labels_3d'].
        meta_keys (list[str]): Keys to save into data_samples.metainfo for each frame.
    """
    def __init__(self,
                 keys: tuple,
                 meta_keys: tuple = ('img_path', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'num_pts_feats', 'pcd_trans',
                            'sample_idx', 'pcd_scale_factor', 'pcd_rotation',
                            'pcd_rotation_angle', 'lidar_path',
                            'transformation_3d_flow', 'trans_mat',
                            'affine_aug', 'sweep_img_metas', 'ori_cam2img',
                            'cam2global', 'crop_offset', 'img_crop_offset',
                            'resize_img_shape', 'lidar2cam', 'ori_lidar2img',
                            'num_ref_frames', 'num_views', 'ego2global',
                            'axis_align_matrix')):
        super().__init__(keys, meta_keys)


    def transform(self, results: Union[dict, List[dict]]) -> Union[dict, List[dict]]:
        assert 'multi_frame_data' in results, 'multi_frame_data are not found in results'
        assert isinstance(results['multi_frame_data'], OrderedDict), 'multi_frame_data must be an OrderedDict'

        ## pack gt
        packed_results = super().pack_single_results(results)

        ## pack inputs
        multi_frame_inputs = []
        multi_frame_metainfos = OrderedDict()
        for k, value in results['multi_frame_data'].items():
            single_frame_result = super().pack_single_results(value)
            multi_frame_inputs.append(single_frame_result['inputs'])
            multi_frame_metainfos[k] = single_frame_result['data_samples'].metainfo
        packed_results['data_samples'].set_field(multi_frame_metainfos, 'temporal_metainfos')
        packed_results['data_samples'].set_metainfo(multi_frame_metainfos[0])

        ## pack inputs
        if 'img' in self.keys:
            imgs = [frame['img'] for frame in multi_frame_inputs]
            packed_results['inputs']['img'] = torch.stack(imgs, 0)
        if 'points' in self.keys:
            points = [frame['points'] for frame in multi_frame_inputs]
            packed_results['inputs']['points'] = points

        return packed_results