import copy
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch import Tensor
from mmdet.models.dense_heads import DETRHead
from mmdet.utils import OptConfigType, ConfigType, OptMultiConfig
from mmdet.models.layers.transformer import inverse_sigmoid
from mmdet3d.utils.typing_utils import InstanceList, SampleList
from mmengine.structures import InstanceData
from mmengine.model import bias_init_with_prob

from mmhdmap.registry import MODELS, TASK_UTILS


@MODELS.register_module()
class BEVFormerHead(DETRHead):

    def __init__(self,
                 num_classes: int = 10,
                 embed_dims: int = 256,
                 num_reg_fcs: int = 2,
                 num_cls_fcs: int = 2,
                 num_pred_layers: int = 6,
                 bev_h: int = 30,
                 bev_w: int = 30,
                 with_box_refine=False,
                 as_two_stage=False,
                 bbox_coder=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg: OptMultiConfig = None,
                 **kwargs) -> None:
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_reg_fcs = num_reg_fcs
        self.num_cls_fcs = num_cls_fcs
        self.num_pred_layers = num_pred_layers
        self.with_box_refine = with_box_refine
        self.code_size = 10

        super().__init__(
            num_classes=num_classes,
            embed_dims=embed_dims,
            num_reg_fcs=num_reg_fcs,
            init_cfg=init_cfg,
            **kwargs
        )

        self.bbox_coder = TASK_UTILS.build(bbox_coder)

    def _init_layers(self):

        cls_branch = []
        for _ in range(self.num_cls_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)
    
        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        if self.with_box_refine:
            self.cls_branches = _get_clone(fc_cls, self.num_pred_layers)
            self.reg_branches = _get_clone(reg_branch, self.num_pred_layers)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(self.num_pred_layers)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(self.num_pred_layers)])
        
    def init_weights(self):
        
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m[-1].bias, bias_init)

    def forward(self,
                hidden_states: Tensor,
                references: List[Tensor]) -> Tuple[Tensor, Tensor]:
        
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []

        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state)
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)

            assert reference.shape[-1] == 3
            tmp_reg_preds[..., 0:2] += reference[..., 0:2]
            tmp_reg_preds[..., 0:2] = tmp_reg_preds[..., 0:2].sigmoid()
            tmp_reg_preds[..., 4:5] += reference[..., 2:3]
            tmp_reg_preds[..., 4:5] = tmp_reg_preds[..., 4:5].sigmoid()

            outputs_coord = tmp_reg_preds
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)
        
        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)

        preds_dicts = dict(
            all_cls_scores = all_layers_outputs_classes,
            all_bbox_preds = all_layers_outputs_coords
        )
        return preds_dicts


    def predict(self,
                hidden_states: Tensor,
                references: List[Tensor],
                batch_data_samples: SampleList) -> InstanceList:
        
        batch_sample_metas = [
            data_samples.metainfo
            for data_samples in batch_data_samples
        ]

        preds_dicts = self(hidden_states, references)

        predictions = self.get_bboxes(preds_dicts, batch_sample_metas)
        
        return predictions

    
    def get_bboxes(self, preds_dicts, batch_metainfos):

        preds_dicts = self.bbox_coder.decode(preds_dicts)
        
        num_samples = len(preds_dicts)
        results_list = []
        for i in range(num_samples):
            bboxes = preds_dicts[i]['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            code_size = bboxes.shape[-1]

            results = InstanceData()
            results.bboxes = batch_metainfos[i]['box_type_3d'](bboxes, code_size)
            results.scores = preds_dicts[i]['scores']
            results.labels = preds_dicts[i]['labels']
        
            results_list.append(results)
        
        return results_list



        