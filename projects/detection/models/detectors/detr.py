from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mmdet.structures import OptSampleList
from mmdet.models.layers import (DetrTransformerDecoder, DetrTransformerEncoder)
from ..layers import SinePositionalEncoding
from .base_detr import DetectionTransformer

from mmhdmap.registry import MODELS


@MODELS.register_module()
class DETR(DetectionTransformer):

    def _init_layers(self) -> None:
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DetrTransformerEncoder(**self.encoder)
        self.decoder = DetrTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims

        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims
    
    def init_weight(self) -> None:
        super().init_weights()
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
    
    def pre_transformer(
        self,
        img_feats: Tuple[Tensor],
        batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:

        feat = img_feats[-1]
        batch_size, feat_dim, _, _ = feat.shape
        
        assert batch_data_samples is not None
        batch_input_shape = batch_data_samples[0].batch_input_shape
        input_img_h, input_img_w = batch_input_shape
        img_shape_list = [sample.img_shape for sample in batch_data_samples]
        same_shape_flag = all([
            s[0] == input_img_h and s[1] == input_img_w for s in img_shape_list
        ])
        if torch.onnx.is_in_onnx_export() or same_shape_flag:
            masks = None
            pos_embed = self.positional_encoding(masks, input=feat)
        else:
            masks = feat.new_ones((batch_size, input_img_h, input_img_w))
            for img_id in range(batch_size):
                img_h, img_w = img_shape_list[img_id]
                masks[img_id, :img_h, :img_w] = 0
            
            masks = F.interpolate(
                masks.unsqueeze(1),
                size=feat.shape[-2:]).to(torch.bool).squeeze(1)
            
            pos_embed = self.positional_encoding(masks)
        
        feat = feat.view(batch_size, feat_dim, -1).permute(0, 2, 1)
        pos_embed = pos_embed.view(batch_size, feat_dim, -1).permute(0, 2, 1)

        if masks is not None:
            masks = masks.view(batch_size, -1)
        
        encoder_inputs_dict = dict(
            feat=feat, feat_mask=masks, feat_pos=pos_embed)
        decoder_inputs_dict = dict(
            memory_mask=masks, memory_pos=pos_embed)

        return encoder_inputs_dict, decoder_inputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor) -> Dict:
        memory = self.encoder(
            query=feat, query_pos=feat_pos,
            key_padding_mask=feat_mask)
        encoder_outputs_dict = dict(memory=memory)
        
        return encoder_outputs_dict
    
    def pre_decoder(self, memory: Tensor) -> Tuple[Dict, Dict]:
        batch_size = memory.size(0)
        query_pos = self.query_embedding.weight
        query_pos = query_pos.unsqueeze(0).repeat(batch_size, 1, 1)
        query = torch.zeros_like(query_pos)

        decoder_inputs_dict = dict(
            query_pos=query_pos, query=query, memory=memory)
        head_inputs_dict=dict()
        
        return decoder_inputs_dict, head_inputs_dict
    
    def forward_decoder(self, query: Tensor, query_pos: Tensor, memory: Tensor,
                        memory_mask: Tensor, memory_pos: Tensor) -> Dict:
        hidden_states = self.decoder(
            query=query,
            key=memory,
            value=memory,
            query_pos=query_pos,
            key_pos=memory_pos,
            key_padding_mask=memory_mask)
        
        head_inputs_dict = dict(hidden_states=hidden_states)
        return head_inputs_dict
    