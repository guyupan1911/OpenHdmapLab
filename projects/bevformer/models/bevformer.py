from typing import Optional, Union, List, Tuple, Dict

import torch
from torch import Tensor
import torch.nn as nn

from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.structures.det3d_data_sample import (Det3DDataSample, SampleList,
                                                  OptSampleList, ForwardResults)
from mmdet3d.utils.typing_utils import InstanceList
from mmdet.utils import OptConfigType, ConfigType, OptMultiConfig

from mmhdmap.registry import MODELS


@MODELS.register_module()
class BEVFormer(Base3DDetector):

    def __init__(self,
                 img_backbone: ConfigType,
                 img_neck: OptConfigType = None,
                 bev_encoder: OptConfigType = None,
                 decoder: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 positional_encoding: OptConfigType = None,
                 with_box_refine: bool = False,
                 as_two_stage: bool = False,
                 embed_dims: int = 256,
                 num_feature_levels: int = 4,
                 bev_h: int = 30,
                 bev_w: int = 30,
                 num_query: int = 900,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 video_test_mode: bool = False,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 **kwargs) -> None:
        
        super().__init__(
            data_preprocessor=data_preprocessor,
            init_cfg = init_cfg)
        
        self.embed_dims = embed_dims
        self.num_query = num_query
        self.bev_h = bev_h
        self.bev_w = bev_w

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.bev_encoder = bev_encoder
        self.decoder = decoder
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage

        # TODO: use grid mask

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

        # init layers
        self.img_backbone = MODELS.build(img_backbone)
        if img_neck is not None:
            self.img_neck = MODELS.build(img_neck)
        self.bbox_head = MODELS.build(bbox_head)
        self.positional_encoding = MODELS.build(positional_encoding)
        self._init_layers()
    
    def _init_layers(self) -> None:

        self.bev_encoder = MODELS.build(self.bev_encoder)
        self.decoder = MODELS.build(self.decoder)

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = nn.Embedding(
                self.num_query, self.embed_dims * 2)
        
        self.reference_points = nn.Linear(self.embed_dims, 3)

    def extract_img_feat(self, batch_inputs: Tensor) -> List[Tensor]:
        """
        Args:
            batch_inputs (Tensor): Image tensor, has shape
                (bs, T, num_cams, C_in, H, W). T is temporal queue and
                T[-1] is current frame.

        Returns:
            List[Tensor]: Multi level feature maps, each has shape
                (bs, T, num_cams, C_out, H, W)
        """
        assert batch_inputs.dim() == 6
        bs, T, num_cams, dim, H, W = batch_inputs.shape
        batch_inputs = batch_inputs.view(bs * T * num_cams, dim, H, W)

        x = self.img_backbone(batch_inputs)
        if self.img_neck is not None:
            img_feats = self.img_neck(x)
        
        multi_level_img_feats = []
        for img_feat in img_feats:
            _, C, H, W = img_feat.shape
            img_feat_reshape = img_feat.view(bs, T, num_cams, C, H, W)
            multi_level_img_feats.append(img_feat_reshape)
        
        return multi_level_img_feats

    def extract_feat(self, batch_inputs_dict: dict):
        assert 'imgs' in batch_inputs_dict
        return self.extract_img_feat(batch_inputs_dict['imgs'])

    def loss(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList) -> Union[dict, tuple]:
        """
        Args:

        Returns:
            dict: A dictionary of loss component
        """


        img_feats = self.extract_img_feat(batch_inputs)
        head_inputs_dict = self.forward_bev_encoder(img_feats,
                                                    batch_data_samples)
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        
        return losses
        
    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        """
        Args:
            batch_inputs (Tensor): Image tensor, has shape
                (bs, T, num_cams, C_in, H, W). T is temporal queue and
                T[-1] is current frame.
        
        Results:
            List[Det3dDataSample]: Detection results of input
        """
        
        # TODO: use history bev

        # only use current frame   
        mlvl_img_feats = self.extract_feat(batch_inputs)
        
        bev_encoder_outputs = self.forward_bev_encoder(mlvl_img_feats, batch_data_samples)

        decoder_outputs_dict = self.forward_decoder(bev_encoder_outputs['bev_embed'])

        hidden_states = decoder_outputs_dict['hidden_states']
        references = decoder_outputs_dict['references']

        print(f'hidden_states: {hidden_states.shape}')
        print(f'references: {len(references)}')

        results_list_3d = self.bbox_head.predict(
            **decoder_outputs_dict,
            batch_data_samples=batch_data_samples)

        print(f'results_list: {results_list_3d}')

        detsamples = self.add_pred_to_datasample(
            batch_data_samples, data_instances_3d=results_list_3d)
        
        return detsamples

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        img_feats = self.extract_img_feat(batch_inputs)
        head_inputs_dict = self.forward_bev_encoder(img_feats,
                                                    batch_data_samples)
        results = self.bbox_head.forward(**head_inputs_dict)
        return results

    def forward_bev_encoder(self, mlvl_feats, batch_data_samples) -> Tensor:
        """
        Args:

        Returns:
            bev_embedding: bev features after temporal self attn and spatial cross attn,
                has shape (bs, bev_h*bev_w, embed_dims)
        """
        mlvl_feats = [feat[:, -1] for feat in mlvl_feats] # current frame
        bs, num_cams, _, _, _ = mlvl_feats[0].shape

        bev_query = self.bev_embedding.weight # (bev_h*bev_w, embed_dims)
        bev_mask = bev_query.new_zeros((bs, self.bev_h, self.bev_w)) 
        bev_pos = self.positional_encoding(bev_mask) # (1, embed_dims, bev_h, bev_w)

        print(f'bev_query: {bev_query.shape}')
        print(f'bev_pos: {bev_pos.shape}')

        # -> (bev_h*bev_w, bs, embed_dims)
        bev_query = bev_query.unsqueeze(1).repeat(1, bs, 1)
        # -> (bev_h*bev_w, 1, embed_dims)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)


        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cams, C, H, W = feat.shape
            spatial_shape = (H, W)
            # -> (num_cams, bs, H*W, C)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
        
        # -> (num_cams, sum(HW), bs, embed_dims)
        feat_flatten = torch.cat(feat_flatten, 2)
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)

        # -> (num_levels, 2）
        spatial_shapes = bev_pos.new_tensor(spatial_shapes, dtype=torch.long)
        level_sizes = spatial_shapes.prod(dim=1)
        level_start_index = torch.cat([level_sizes.new_zeros(1), level_sizes.cumsum(0)[:-1]])

        bev_embed = self.bev_encoder(
            bev_query=bev_query,
            key=feat_flatten,
            value=feat_flatten,
            bev_pos=bev_pos,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=None,
            shift=None,
            batch_data_samples=batch_data_samples)
        
        return {
            "bev_embed": bev_embed
        }

    def forward_decoder(self, bev_embed) -> Tensor:
        # bev_embed: (bs, bev_h*bev_w, embed_dims)
        bs = bev_embed.size(0)

        # object query embeddings: (num_query, embed_dims * 2)
        object_query_embedding = self.query_embedding.weight
        query_pos, query = torch.split(
            object_query_embedding, self.embed_dims, dim=1)

        # expand to batch-first format expected by DeformableDetrTransformerDecoderLayer
        # query_pos / query: (bs, num_query, embed_dims)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)

        # reference points on BEV plane, shape: (bs, num_query, 3)
        reference_points = self.reference_points(query_pos).sigmoid()

        init_reference_out = reference_points

       
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=None,
            cls_branches=None,
            # MultiScaleDeformableAttention expects Long (int64) for shapes
            spatial_shapes=query.new_tensor([[self.bev_h, self.bev_w]],
                                            dtype=torch.long),
            level_start_index=query.new_tensor([0], dtype=torch.long)
        )
        
        references = [reference_points, *inter_references]
        decoder_outputs_dict = dict(
            hidden_states = inter_states,
            references = references
        )

        return decoder_outputs_dict