from typing import Dict, Optional, Tuple, Union, List

import torch
from torch import nn, Tensor

from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.structures.bbox_3d.utils import get_lidar2img
from mmdet3d.structures.det3d_data_sample import (ForwardResults,
                                                  OptSampleList, SampleList)
from mmdet3d.utils.typing_utils import (OptConfigType, OptInstanceList,
                                        OptMultiConfig)
from mmengine.structures import InstanceData
from mmengine.optim import OptimWrapper

from .detr3d_head import DETR3DHead
from .grid_mask import GridMask

class DETR3D(nn.Module):
    def __init__(self,
                 img_backbone: Optional[dict] = None,
                 img_neck: Optional[dict] = None,
                 data_preprocessor=None,
                 pts_bbox_head=None,
                 train_cfg=None,
                 use_grid_mask=False,
                 **kwargs):
        super().__init__()
        # data preprocessor
        self.data_preprocessor = MODELS.build(data_preprocessor)
        self.data_preprocessor.to('cuda')

        # img backbone and neck
        self.img_backbone = MODELS.build(img_backbone)
        self.img_neck = MODELS.build(img_neck)
        
        # decoder
        pts_train_cfg = train_cfg.pts
        pts_bbox_head.update(train_cfg=pts_train_cfg)
        self.pts_bbox_head = DETR3DHead(**pts_bbox_head)

        # train_cfg
        self.train_cfg = train_cfg

        # grid mask
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask


    def extract_img_feat(self, img: Tensor,
                         batch_input_metas: List[Dict]) -> List[Tensor]:
        """
            Args:
                img: torch.Tensor (bs, num_cams, C, H, W)
                batch_input_metas: list of datasample's metainfo
            Returns:
                img_feats: multi_level image features

        """
        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]
            for img_meta in batch_input_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        
        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        
        return img_feats_reshaped


    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor'):
        """
            Args:
                inputs: {'imgs'}
                    imgs: torch.Tensor (bs, num_cams, C, H, W)
                data_samples: list of Det3DDataSample
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            pass


    def val_step(self, data: dict):
        """
            Args:
                data: {'data_samnples', 'inputs'}
                    inputs: {'img'}
                        img: list of Tensor
                    data_samples: list of Det3DDataSample
        """
  
        data = self.data_preprocessor(data, False)
        results = self(**data, mode='predict')

        return results


    def train_step(self, data, optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self(**data, mode='loss')
        parsed_losses, log_vars = self.parse_losses(losses)
        optim_wrapper.update_params(parsed_losses)
        return log_vars


    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        batch_input_metas = self.add_lidar2img(batch_input_metas)
        
        # extract multi_level image features
        img_feats = self.extract_img_feat(batch_inputs_dict['imgs'], batch_input_metas)

        #  decoder
        outs = self.pts_bbox_head(img_feats, batch_input_metas, **kwargs)

        # loss
        batch_gt_instances_3d = [
           item.gt_instances_3d for item in batch_data_samples
        ]
        loss_inputs = [batch_gt_instances_3d, outs]
        losses_pts = self.pts_bbox_head.loss_by_feat(*loss_inputs)

        return losses_pts


    def predict(self,
                batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs):
        """
            Args:
                batch_inputs_dict: {'imgs'}
                    imgs: torch.Tensor (bs, num_cams, C, H, W
                batch_data_samples: list of Det3DDataSample
        """

        batch_input_metas = [item.metainfo for item in batch_data_samples]
        batch_input_metas = self.add_lidar2img(batch_input_metas)
        for i in range(len(batch_data_samples)):
            batch_data_samples[i].set_metainfo(batch_input_metas[i])
            # print(batch_data_samples[i].metainfo['lidar2img'])
        # extract multi-level image features (4)
        img_feats = self.extract_img_feat(batch_inputs_dict['imgs'], batch_input_metas)

        # decoder
        outs = self.pts_bbox_head(img_feats, batch_input_metas)
        
        # ouputs: dict
        # all_cls_scores: torch.Tensor (num_layers, bs, num_query, classes)
        # all_bbox_preds: torch.Tensor (num_layers, bs, num_query, codes)

        # predict
        results_list_3d = self.pts_bbox_head.predict_by_feat(
            outs, batch_input_metas, **kwargs)
        
        datasamples = self.add_pred_to_datasample(batch_data_samples, results_list_3d)

        return datasamples


    def add_lidar2img(self, batch_input_metas: List[Dict]) -> List[Dict]:
        """add 'lidar2img' transformation matrix into batch_input_metas.

        Args:
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.

        Returns:
            batch_input_metas (list[dict]): Meta info with lidar2img added
        """
        for meta in batch_input_metas:
            l2i = list()
            for i in range(len(meta['cam2img'])):
                c2i = torch.tensor(meta['cam2img'][i]).double()
                l2c = torch.tensor(meta['lidar2cam'][i]).double()
                l2i.append(get_lidar2img(c2i, l2c).float().numpy())
            meta['lidar2img'] = l2i
        return batch_input_metas


    def add_pred_to_datasample(
        self,
        data_samples: SampleList,
        data_instances_3d: OptInstanceList = None,
        data_instances_2d: OptInstanceList = None,
    ) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Subclasses could override it to be compatible for some multi-modality
        3D detectors.

        Args:
            data_samples (list[:obj:`Det3DDataSample`]): The input data.
            data_instances_3d (list[:obj:`InstanceData`], optional): 3D
                Detection results of each sample.
            data_instances_2d (list[:obj:`InstanceData`], optional): 2D
                Detection results of each sample.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input. Each Det3DDataSample usually contains
            'pred_instances_3d'. And the ``pred_instances_3d`` normally
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels_3d (Tensor): Labels of 3D bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (Tensor): Contains a tensor with shape
              (num_instances, C) where C >=7.

            When there are image prediction in some models, it should
            contains  `pred_instances`, And the ``pred_instances`` normally
            contains following keys.

            - scores (Tensor): Classification scores of image, has a shape
              (num_instance, )
            - labels (Tensor): Predict Labels of 2D bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Contains a tensor with shape
              (num_instances, 4).
        """

        assert (data_instances_2d is not None) or \
               (data_instances_3d is not None),\
               'please pass at least one type of data_samples'

        if data_instances_2d is None:
            data_instances_2d = [
                InstanceData() for _ in range(len(data_instances_3d))
            ]
        if data_instances_3d is None:
            data_instances_3d = [
                InstanceData() for _ in range(len(data_instances_2d))
            ]

        for i, data_sample in enumerate(data_samples):
            data_sample.pred_instances_3d = data_instances_3d[i]
            data_sample.pred_instances = data_instances_2d[i]
        return data_samples
