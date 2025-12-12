from typing import Optional, Union, List

import torch
from torch import Tensor
import torch.nn as nn

from mmengine.model import BaseModel
from mmdet3d.structures.det3d_data_sample import (Det3DDataSample, SampleList,
                                                  OptSampleList, ForwardResults)
from mmdet3d.utils.typing_utils import InstanceList
from mmdet.utils import OptConfigType, ConfigType, OptMultiConfig

from mmhdmap.registry import MODELS


@MODELS.register_module()
class BEVFormer(BaseModel):

    def __init__(self,
                 img_backbone: ConfigType,
                 img_neck: OptConfigType = None,
                 encoder: OptConfigType = None,
                 decoder: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 positional_encoding: OptConfigType = None,
                 with_box_refine: bool = False,
                 num_feature_levels: int = 4,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 video_test_mode: bool = False,
                 init_cfg: OptMultiConfig = None,
                 **kwargs) -> None:
        
        super().__init__(
            data_preprocessor=data_preprocessor,
            init_cfg = init_cfg)
        
        # bbox_head.update(train_cfg=train_cfg)
        # bbox_head.update(test_cfg=test_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.encoder = encoder
        self.decoder = decoder
        self.positional_encoding = positional_encoding

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
        # self.bbox_head = MODELS.build(bbox_head)
        self._init_layers()
    
    def _init_layers(self) -> None:

        self.encoder = MODELS.build(self.encoder)
        self.decoder = MODELS.build(self.decoder)


    def forward(self,
                inputs: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".'
                                'Only support loss, predict and tensor mode')
    
    def loss(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList) -> Union[dict, tuple]:
        """
        Args:

        Returns:
            dict: A dictionary of loss component
        """


        img_feats = self.extract_img_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
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
        img_feats = self.extract_img_feat(batch_inputs[:,-1].unsqueeze(1))

        head_inputs_dict = self.forward_transformer(img_feats, batch_data_samples)

        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            batch_data_samples=batch_data_samples)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        
        return batch_data_samples

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        img_feats = self.extract_img_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        results = self.bbox_head.forward(**head_inputs_dict)
        return results
        
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
        

    def forward_transformer(self,
                            img_feats: Tuple[Tensor],
                            batch_data_samples: OptSampleList = None) -> Dict:
        """
        Args:
            img_feats: (tuple[Tensor]): Tuple of feature maps from neck, each
                has shape (bs, T, num_cams, dim, H, W)
            batch_data_samples: (list[Det3dDataSample])

        Returns:
            dict: A dictionary of bbox_head function inputs
        """

        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)
        
        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(**encoder_outputs_dict)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)

        return head_inputs_dict

    def pre_transformer(self,
                        img_feats: Tuple[Tensor],
                        batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:
        """
        prepare inputs for bevformer encoder, process canbus

        Args:
            img_feats: multi level img feats
        
        Returns:
            tuple[dict, dict]: The first dict contains the inputs of encoder
            and the second dict contains the inputs of decoder
        """

    def forward_encoder(self,
                        feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, **kwargs) -> Dict:
        pass

    def pre_decoder(self, memory: Tensor, **kwargs) -> Tuple[Dict, Dict]:
        pass

    def forward_decoder(self, query: Tensor, query_pos: Tensor, memory: Tensor,
                        **kwargs) -> Dict:
        pass

    def add_pred_to_datasample(self,
                                data_samples: SampleList,
                                results_list: InstanceList) -> SampleList:
        pass
    