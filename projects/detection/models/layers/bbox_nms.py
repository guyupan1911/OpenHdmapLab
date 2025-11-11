from typing import Optional, Tuple, Union


import torch
from mmcv.ops.nms import batched_nms
from torch import Tensor

from mmdet.utils import ConfigType


def multiclass_nms(
    multi_bboxes: Tensor,
    multi_scores: Tensor,
    score_thr: float,
    nms_cfg: ConfigType,
    max_num: int = -1,
    score_factors: Optional[Tensor] = None,
    return_inds: bool = False,
    box_dim: int = 4
) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor]]:
    
    num_classes = multi_scores.size(1) - 1
    if multi_bboxes.shape[1] > box_dim:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, box_dim)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, box_dim)
    

    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, box_dim)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    if not torch.onnx.is_in_onnx_export():
        valid_mask = scores > score_thr
    
    if score_factors is not None:
        score_factors = score_factors.view(-1, 1).expand(
            multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors
    
    if not torch.onnx.is_in_onnx_export():
        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    else:
        bboxes = torch.cat([bboxes, bboxes.new_zeros(1, box_dim)], dim=0)
        scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
        labels = torch.cat([labels, labels.new_zeros(1)], dim=0)

    if bboxes.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        dets = torch.cat([bboxes, scores[:, None]], -1)
        if return_inds:
            return dets, labels, inds
        else:
            return dets, labels
        
    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]
    
    if return_inds:
        return dets, labels[keep], inds[keep]
    else:
        return dets, labels[keep]

