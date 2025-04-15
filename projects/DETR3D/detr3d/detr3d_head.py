import copy
from typing import Dict, List, Tuple

import torch
from torch import nn, Tensor
from mmcv.cnn import Linear, Conv2d, build_activation_layer
from mmdet.models.layers import inverse_sigmoid
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean
from mmengine.structures import InstanceData

from .detr3d_transformer import Detr3DTransformer

class DETR3DHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query=100,
                 sync_cls_avg_factor=True,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 positional_encoding=None,
                 loss_cls=None,
                 loss_bbox=None,
                 loss_iou=None,
                 train_cfg=None,
                 test_cfg=None,
                 num_reg_fcs=2,
                 num_cls_fcs=2,
                 code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
                 code_size=10,
                 **kwargs):
        super().__init__()
        # parameters
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        self.code_size = code_size
        self.code_weights = code_weights

        # DETR3D transformer
        self.transformer = Detr3DTransformer(**transformer)
        self.embed_dims = self.transformer.embed_dims

        # bbox_coder
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range

        # task
        self.num_cls_fcs = num_cls_fcs - 1

        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None:
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            class_weight = torch.ones(num_classes + 1) * class_weight
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assigner = train_cfg['assigner']
            self.assigner = TASK_UTILS.build(assigner)

        # loss
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_iou = MODELS.build(loss_iou)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.act_cfg = transformer.get('act_cfg', dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)
        self.positional_encoding = MODELS.build(positional_encoding)
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims

        # sampler
        sampler_cfg = dict(type='PseudoSampler')
        self.sampler = TASK_UTILS.build(sampler_cfg)

        # coder
        self.code_weights = nn.Parameter(
            torch.tensor(self.code_weights, requires_grad=False),
            requires_grad=False)

        self._init_layers()

    def _init_layers(self) -> None:

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList([
                fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([
                reg_branch for _ in range(num_pred)])
        
        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)


    def forward(self, mlvl_feats: List[Tensor], img_metas: List[Dict],
                **kwargs) -> Dict[str, Tensor]:
        """
            Args:
                mlvl_feats: multi_level image features 4
                img_metas: list of metainfo
        """

        # (num_query, embed_dims*2)
        query_embeds = self.query_embedding.weight
        hs, init_reference, inter_references = self.transformer(
            mlvl_feats,
            query_embeds,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            img_metas=img_metas,
            **kwargs)
        
        # hs (num_layers, num_query, bs, C)
        # init_references (bs, num_query, 3)
        # inter_references (num_layer, bs, num_query, 3)

        # -> (num_layers, bs, num_query, C)
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            # (bs, num_query, classes)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            # (bs, num_query, 10)
            tmp = self.reg_branches[lvl](hs[lvl])  # shape: ([B, num_q, 10])
            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            tmp[..., 0:1] = \
                tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) \
                + self.pc_range[0]
            tmp[..., 1:2] = \
                tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) \
                + self.pc_range[1]
            tmp[..., 4:5] = \
                tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) \
                + self.pc_range[2]

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outs = {
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }
        return outs

    def predict_by_feat(self,
                        preds_dicts,
                        img_metas,
                        rescale=False) -> InstanceList:
        """
            Args:
            pred_dicts: dict{'all_cls_scores', 'all_bbox_preds'}
                all_cls_scores: torch.Tensor (num_layers, bs, num_query, classes)
                all_bbox_predsL torch.Tensor (num_layers, bs, num_query, codes)
        """

        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)  # batch size
        ret_list = []
        for i in range(num_samples):
            results = InstanceData()
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, self.code_size - 1)

            results.bboxes_3d = bboxes
            results.scores_3d = preds['scores']
            results.labels_3d = preds['labels']
            ret_list.append(results)
        return ret_list


    def loss_by_feat(self, batch_gt_instances_3d, preds_dicts) -> Dict:
        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_preds = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        batch_gt_instances_3d_list = [
           batch_gt_instances_3d for _ in range(num_dec_layers)
        ]
        losses_cls, losses_bbox = multi_apply(self.loss_by_feat_single,
                                              all_cls_scores, all_bbox_preds,
                                              batch_gt_instances_3d_list)
        
        loss_dict = dict()

        if enc_cls_scores is not None:
            enc_loss_cls, enc_losses_bbox = self.loss_by_feat_single(
                enc_cls_scores, enc_bbox_preds, batch_gt_instances_3d_list)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]


        num_dec_layers = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict