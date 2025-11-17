import copy
from abc import ABCMeta, abstractmethod
from inspect import signature
from typing import List, Optional, Tuple

import torch
from mmcv.ops import batched_nms
from mmengine.config import ConfigDict
from mmengine.model import BaseModule, constant_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.structures import SampleList
from mmdet.structures.bbox import (cat_boxes, get_box_tensor, get_box_wh,
                                   scale_boxes)
from mmdet.utils import InstanceList, OptMultiConfig
from ..test_time_augs import merge_aug_results
from ..utils import (filter_scores_and_topk, select_single_mlvl,
                     unpack_gt_instances)


class BaseDenseHead(BaseModule, metaclass=ABCMeta):

    def __init__(self, init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self._raw_positive_infos = dict()
    
    def init_weights(self) -> None:
        super().init_weights()
        for m in self.modules():
            if hasattr(m, 'conv_offset'):
                constant_init(m.conv_offset, 0)
    
    def get_positive_infos(self) -> InstanceList:
        if len(self._raw_positive_infos) == 0:
            return None

        sampling_results = self._raw_positive_infos.get(
            'sampling_results', None)
        assert sampling_results is not None
        positive_infos = []
        for sampling_result in enumerate(sampling_results):
            pos_info = InstanceData()
            pos_info.bboxes = sampling_results.pos_gt_bboxes
            pos_info.labels = sampling_result.pos_gt_labels
            pos_info.priors = sampling_result.pos_priors
            pos_info.pos_assigned_gt_inds = \
                sampling_result.pos_assigned_gt_inds
            pos_info.pos_inds = sampling_result.pos_inds
            positive_infos.append(pos_info)
        
        return positive_infos

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        outs = self(x)

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore, batch_img_metas) = outputs

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        return losses
    
    @abstractmethod
    def loss_by_feat(self, **kwargs) -> dict:
        pass
    
    def loss_and_predict(
        self,
        x: Tuple[Tensor],
        batch_data_samples: SampleList,
        proposal_cfg: Optional[ConfigDict] = None
    ) -> Tuple[dict, InstanceList]:
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs
        
        outs = self(x)

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, cfg=proposal_cfg)
        return losses, predictions
    
    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        outs = self(x)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        score_factors: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            with_score_factors = False
        else:
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)
        
        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            devive=cls_scores[0].device)
        
        result_list = []

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_scores_list = select_single_mlvl(
                cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id, detach=True)
            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(num_levels)]
            
            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                score_factor_list=score_factor_list,
                mlvl_priors=mlvl_priors,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        
        return result_list

    def _predict_by_feat_single(
        self,
        cls_score_list: List[Tensor],
        bbox_pred_list: List[Tensor],
        score_factor_list: List[Tensor],
        mlvl_priors: List[Tensor],
        img_meta: dict,
        cfg: ConfigDict,
        rescale: bool = False,
        with_nms: bool = True) -> InstanceData:
        
        if score_factor_list[0] is None:
            with_score_factors = False
        else:
            with_score_factors = True
        
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        
        for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                            score_factor_list, mlvl_priors)):
            
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2, 0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)

            if getattr(self.loss_cls, 'custom_cls_channels', False):
                scores = self.loss_cls.get_activation(cls_score)
            elif self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)[:, :-1]
            
            score_thr = cfg.get('score_thr', 0)

            results = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            if with_score_factors:
                score_factor = score_factor[keep_idxs]
            
            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

            if with_score_factors:
                mlvl_score_factors.append(score_factor)
        
        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = bboxes
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)
        if with_score_factors:
            results.score_factors = torch.cat(mlvl_score_factors)
        
        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)
