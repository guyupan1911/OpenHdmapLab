from typing import Optional

import numpy as np
import torch
from mmengine.dist import master_only
from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import Det3DDataSample

from mmhdmap.registry import VISUALIZERS


@VISUALIZERS.register_module()
class MultiFrameDet3DLocalVisualizer(Det3DLocalVisualizer):

    @master_only
    def add_datasample(self,
                       name: str,
                       data_input: dict,
                       data_sample: Optional[Det3DDataSample] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       show: bool = False,
                       wait_time: float= 0,
                       out_file: Optional[str] = None,
                       o3d_save_path: Optional[str] = None,
                       vis_task: str = 'mono_det',
                       pred_score_thr: float = 0.3,
                       step: int = 0,
                       show_pcd_rgb: bool = False) -> None:
        
        # inputs
        new_data_input = {}
        new_data_input['img'] = data_input['img'][-1]
        new_data_input['points'] = data_input['points'][-1]

        # data_samples
        new_data_sample = Det3DDataSample()
        new_data_sample.gt_instances_3d = data_sample.gt_instances_3d
        
        # Fix yaw angle for prediction instances: convert from nuscenes format to mmdet3d format
        # In nuscenes, yaw is defined differently than in mmdet3d visualization
        # Need to negate yaw angles for correct visualization in mmdet3d
        if data_sample.pred_instances_3d is not None and hasattr(data_sample.pred_instances_3d, 'bboxes_3d'):
            pred_instances_3d = data_sample.pred_instances_3d.clone()
            if pred_instances_3d.bboxes_3d is not None and len(pred_instances_3d.bboxes_3d) > 0:
                # Get the tensor representation of bboxes
                bboxes_tensor = pred_instances_3d.bboxes_3d.tensor.clone()
                # Negate yaw angles (yaw is at index 6 in [x, y, z, w, l, h, yaw, vx, vy])
                if bboxes_tensor.shape[-1] >= 7:
                    bboxes_tensor[:, 6] = -bboxes_tensor[:, 6]
                    # Create new bboxes_3d with corrected yaw using the same type as original
                    bbox_type = type(pred_instances_3d.bboxes_3d)
                    pred_instances_3d.bboxes_3d = bbox_type(bboxes_tensor, box_dim=bboxes_tensor.shape[-1])
            new_data_sample.pred_instances_3d = pred_instances_3d
        else:
            new_data_sample.pred_instances_3d = data_sample.pred_instances_3d
        
        # metainfo
        new_metainfo = data_sample.metainfo
        new_metainfo['lidar2img'] = np.stack(new_metainfo['lidar2img'])
        new_data_sample.set_metainfo(new_metainfo)

        super().add_datasample(
            name=name,
            data_input=new_data_input,
            data_sample=new_data_sample,
            draw_gt=draw_gt,
            draw_pred=draw_pred,
            show=show,
            wait_time=wait_time,
            out_file=out_file,
            o3d_save_path=o3d_save_path,
            vis_task=vis_task,
            pred_score_thr=pred_score_thr,
            step=step,
            show_pcd_rgb=show_pcd_rgb)