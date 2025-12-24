from typing import List

import torch
from torch import Tensor


def normalize_bbox(bboxes: Tensor, pc_range: List) -> Tensor:

    # Follow original BEVFormer bbox coding:
    # [cx, cy, cz, w, l, h, yaw, vx, vy] ->
    # [cx, cy, log(w), log(l), cz, log(h), sin(yaw), cos(yaw), vx, vy]
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            [cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy], dim=-1)
    else:
        normalized_bboxes = torch.cat(
            [cx, cy, w, l, cz, h, rot.sin(), rot.cos()], dim=-1)
    
    return normalized_bboxes


def denormalize_bbox(normalized_bboxes, pc_range):

    rot_sine = normalized_bboxes[..., 6:7]
    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # In BEVFormer, cx/cy/cz are already in real-world coordinates (meters)
    # at prediction time. Do NOT rescale by pc_range here.
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    w = normalized_bboxes[..., 2:3].exp()
    l = normalized_bboxes[..., 3:4].exp()
    h = normalized_bboxes[..., 5:6].exp()

    if normalized_bboxes.size(-1) > 8:
        vx = normalized_bboxes[..., 8:9]
        vy = normalized_bboxes[..., 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    
    return denormalized_bboxes