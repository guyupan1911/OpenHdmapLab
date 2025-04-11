import copy

import torch
from torch import nn

from mmdet3d.registry import MODELS, TASK_UTILS
from .bev_transformer import PerceptionTransformer


class BEVFormerHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels=256,
                 num_query=100,
                 num_reg_fcs=2,
                 num_cls_fcs=2,
                 sync_cls_avg_factor=False,
                 transformer=None,
                 positional_encoding=None,
                 loss_cls=None,
                 loss_bbox=None,
                 loss_iou=None,
                 train_cfg=None,
                 bev_h=30,
                 bev_w=30,
                 with_box_refine=False,
                 as_two_stage=False,
                 bbox_coder=None,
                 code_size=10,
                 **kwargs):
        super().__init__()

        # detr parameters
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.num_cls_fcs = num_cls_fcs - 1
        self.train_cfg = train_cfg


        # bev parameters
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        self.pc_range = bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.read_h = self.pc_range[4] - self.pc_range[1]
        self.code_size = code_size
        self.code_weights = [1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)

        # modules
        self.act_cfg = transformer.get('act_cfg', dict(type='ReLU', inplace=True))
        self.activate = MODELS.build(self.act_cfg)
        self.positional_encoding = MODELS.build(positional_encoding)
        # self.bbox_coder = TASK_UTILS.build(bbox_coder)

        # transformer
        self.transformer = PerceptionTransformer(**transformer)
        self.embed_dims = self.transformer.embed_dims
        num_feats = positional_encoding['num_feats']
        assert num_feats*2 == self.embed_dims


        # loss
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_iou = MODELS.build(loss_iou)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        # init layers
        self._init_layers()        


    def _init_layers(self):
        cls_branch = []
        for _ in range(self.num_reg_fcs):
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

        num_pred = (self.transformer.decoder.num_layers + 1) if self.as_two_stage \
            else self.transformer.decoder.num_layers
        
        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
        
        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
    

    def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False):
        """
            Args:
                mlvl_feats, list of Tensor with shape (bs, queue, num_cams, C, H, W)
        """

        # prepare inputs
        bs, Q, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        # forward transformer
        # if only_bev:
