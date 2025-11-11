from typing import List, Optional, Tuple, Union

from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.utils import _pair

# from mmdet.models.layers import multiclass_nms
from projects.detection.models.layers import multiclass_nms
from mmdet.models.utils import empty_instances
from mmdet.structures.bbox import get_box_tensor, scale_boxes
from mmdet.utils import ConfigType, OptMultiConfig, InstanceList

from mmhdmap.registry import MODELS, TASK_UTILS


@MODELS.register_module()
class BBoxHead(BaseModule):

    def __init__(self,
                with_avg_pool: bool = False,
                with_cls: bool = True,
                with_reg: bool = True,
                roi_feat_size: int = 7,
                in_channels: int = 256,
                num_classes: int = 80,
                bbox_coder: ConfigType = dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=True,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                predict_box_type: str = 'hbox',
                reg_class_agnostic: bool = False,
                reg_decoded_bbox: bool = False,
                reg_predictor_cfg: ConfigType = dict(type='Linear'),
                cls_predictor_cfg: ConfigType = dict(type='Linear'),
                loss_cls: ConfigType = dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox: ConfigType = dict(
                    type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.predict_box_type = predict_box_type
        self.reg_class_agnostic = reg_class_agnostic
        self.reg_decoded_bbox = reg_decoded_bbox
        self.reg_predictor_cfg = reg_predictor_cfg
        self.cls_predictor_cfg = cls_predictor_cfg

        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = num_classes + 1
            cls_predictor_cfg_ = self.cls_predictor_cfg.copy()
            cls_predictor_cfg_.update(
                in_features=in_channels, out_features=cls_channels)
            self.fc_cls = MODELS.build(cls_predictor_cfg_)
        if self.with_reg:
            box_dim = self.bbox_coder.encode_size
            out_dim_reg = box_dim if reg_class_agnostic else \
                box_dim * num_classes
            reg_predictor_cfg_ = self.reg_predictor_cfg.copy()
            if isinstance(reg_predictor_cfg_, (dict, ConfigDict)):
                reg_predictor_cfg_.update(
                    in_features=in_channels, out_features=out_dim_reg)
            self.fc_reg = MODELS.build(reg_predictor_cfg_)
        self.debug_imgs = None
        if init_cfg is None:
            self.init_cfg = []
            if self.with_cls:
                self.init_cfg += [
                    dict(type='Normal', std=0.01, override=dict(name='fc_cls'))
                ]
            if self.with_reg:
                self.init_cfg += [
                    dict(type='Normal', std=0.001, override=dict(name='fc_reg'))
                ]
    @property
    def custom_cls_channels(self) -> bool:
        return getattr(self.loss_cls, 'custom_cls_channels', False)

    def forward(self, x: Tuple[Tensor]) -> tuple:
        if self.with_avg_pool:
            if x.numel() > 0:
                x = self.avg_pool(x)
                x = x.view(x.size(0), -1)
            else:
                x = torch.mean(x, dim=(-1, -2))
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred
    
    def predict_by_feat(self,
                        rois: Tuple[Tensor],
                        cls_scores: Tuple[Tensor],
                        bbox_preds: Tuple[Tensor],
                        batch_img_metas: List[dict],
                        rcnn_test_cfg: Optional[ConfigDict] = None,
                        rescale: bool = False) -> InstanceList:
        print(f'rcnn_test_cfg: {rcnn_test_cfg}')
        assert len(cls_scores) == len(bbox_preds)
        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(
                roi=rois[img_id],
                cls_score=cls_scores[img_id],
                bbox_pred=bbox_preds[img_id],
                img_meta=img_meta,
                rescale=rescale,
                rcnn_test_cfg=rcnn_test_cfg)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                roi: Tensor,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                img_meta: dict,
                                rescale: bool = False,
                                rcnn_test_cfg: Optional[ConfigDict] = None) -> InstanceData:
        results = InstanceData()
        if roi.shape[0] == 0:
            return empty_instance([img_meta],
                                  roi.device,
                                  task_type='bbox',
                                  instance_results=[results],
                                  box_type=self.predict_box_type,
                                  use_box_type=False,
                                  num_classes=self.num_classes,
                                  score_per_cls=rcnn_test_cfg is None)[0]

        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(cls_score, dim=-1) if cls_score is not None else None

        img_shape = img_meta['img_shape']
        num_rois = roi.size(0)

        if bbox_pred is not None:
            num_classes = 1 if self.reg_class_agnostic else self.num_classes
            roi = roi.repeat_interleave(num_classes, dim=0)
            bbox_pred = bbox_pred.view(-1, self.bbox_coder.encode_size)
            bboxes = self.bbox_coder.decode(
                roi[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = roi[:, 1:].clone()
            if img_shape is not None and bboxes.size(-1) == 4:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])
        
        if rescale and bboxes.size(0) > 0:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            bboxes = scale_boxes(bboxes, scale_factor)

        bboxes = get_box_tensor(bboxes)
        box_dim = bboxes.size(-1)
        bboxes = bboxes.view(num_rois, -1)

        if rcnn_test_cfg is None:
            results.bboxes = bboxes
            results.scores = scores
        else:
            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                rcnn_test_cfg.score_thr,
                rcnn_test_cfg.nms,
                rcnn_test_cfg.max_per_img,
                box_dim=box_dim)
            results.bboxes = det_bboxes[:, :-1]
            results.scores = det_bboxes[:, -1]
            results.labels = det_labels
        return results
