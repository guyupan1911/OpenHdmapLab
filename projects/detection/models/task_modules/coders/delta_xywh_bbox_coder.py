from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor

from mmhdmap.registry import TASK_UTILS
from .base_bbox_coder import BaseBBoxCoder
from mmdet.structures.bbox import BaseBoxes, HorizontalBoxes, get_box_tensor


@TASK_UTILS.register_module()
class DeltaXYWHBBoxCoder(BaseBBoxCoder):
    def __init__(self,
                 target_means: Sequence[float] = (0., 0., 0., 0.),
                 target_stds: Sequence[float] = (1., 1., 1., 1.),
                 clip_border: bool = True,
                 add_ctr_clamp: bool = False,
                 ctr_clamp: int = 32,
                 **kwargs) -> None:
        super().__init__(**kwargs):
        self.means = target_means
        self.stds = target_stds
        self.clip_border = clip_border
        self.add_ctr_clamp = add_ctr_clamp
        self.ctr_clamp = ctr_clamp

    def encode(self, bboxes: Union[Tensor, BaseBoxes],
               gt_bboxes: Union[Tensor, BaseBoxes]) -> Tensor:
        
        bboxes = get_box_tensor(bboxes)
        gt_bboxes = get_box_tensor(gt_bboxes)
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 4
        encoded_bboxes = bbox2delta(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes
    
    def decode(
        self,
        bboxes: Union[Tensor, BaseBoxes],
        pred_bboxes: Tensor,
        max_shape: Optional[Union[Sequence[int], Tensor,
                                  Sequence[Sequence[int]]]] = None,
        wh_ratio_clip: Optional[float] = 16 / 1000
    ) -> Union[Tensor, BaseBoxes]:
        bboxes = get_box_tensor(bboxes)
        assert pred_bboxes.size(0) == bboxes.size(0)
        if pred_bboxes.ndim == 3:
            assert pred_bboxes.size(1) == bboxes.size(1)
        
        if pred_bboxes.ndim == 2 and not torch.onnx.is_in_onnx_export():
            decoded_bboxes = delta2bbox()
        else:
            decoded_bboxes = oonx_delta2bbox()
        
        if self.use_box_type:
            assert decoded_bboxes.size(-1) == 4
            decoded_bboxes = HorizontalBoxes(decoded_bboxes)
        return decoded_bboxes


def bbox2delta(
    proposals: Tensor, 
    gt: Tensor,
    means: Sequence[float] = (0., 0., 0., 0.),
    stds: Sequence[float] = (1., 1., 1., 1.)
) -> Tensor:
    
    assert proposals.size() == gt.size()
    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0]
    gh = gt[..., 3] - gt[..., 1]

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta2bbox(
    rois: Tensor,
    deltas: Tensor,
    means: Sequence[float] = (0., 0., 0., 0.),
    stds: Sequence[float] = (1., 1., 1., 1.),
    max_shape: Optional[Union[Sequence[int], Tensor,
                              Sequence[Sequence[int]]]] = None,
    wh_ratio_clip: float = 16 / 1000,
    clip_border: bool = True,
    add_ctr_clamp: bool = False,
    ctr_clamp: int = 32) -> Tensor:

    num_bboxes, num_classes = deltas.size(0), deltas.size(1) // 4
