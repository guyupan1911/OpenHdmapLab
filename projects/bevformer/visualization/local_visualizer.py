from typing import Optional

import numpy as np
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