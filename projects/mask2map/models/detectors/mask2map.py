from typing import Dict, List, Tuple, Union

import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmhdmap.registry import MODELS, TASK_UTILS
from mmdet.structures import OptSampleList, SampleList
# from mmdet.structures import BatchHDMapInstance, GroupHDMapInstances
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector


@MODELS.register_module()
class Mask2Map(BaseDetector):

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bev_encoder: OptConfigType = None,
                 bev_neck: OptConfigType = None,
                 mask_decoder: OptConfigType = None,
                 map_decoder: OptConfigType = None,
                 bbox_decoder: OptConfigType = None,
                 mask_head: OptConfigType = None,
                 panoptic_fusion_head: OptConfigType = None,
                 map_head: OptConfigType = None,
                 map_vectorized_head: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 ldaf_head: OptConfigType = None,
                 ldaf_vectorized_head: OptConfigType = None,
                 ldaf_postprocessor: OptConfigType = None,
                 positional_encoding: OptConfigType = None,
                 num_queries: int = 50,
                 num_transformer_feat_level: int = 3,
                 frozen_stages: Union[List[str], Tuple[str]] = (),
                 map_query_generator: OptConfigType = None,
                 mask_denoiser: OptConfigType = None,
                 map_denoiser: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        
        self.bev_encoder_cfg = bev_encoder
        self.bev_neck_cfg = bev_neck
        self.mask_decoder_cfg = mask_decoder
        self.map_decoder_cfg = map_decoder
        self.bbox_decoder_cfg = bbox_decoder
        self.positional_encoding_cfg = positional_encoding
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.map_query_generator_cfg = map_query_generator

        self.mask_denoiser_cfg = mask_denoiser
        self.map_denoiser_cfg = map_denoiser

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if frozen_stages is None:
            frozen_stages = []
        if isinstance(frozen_stages, str):
            frozen_stages = [frozen_stages]
        self.frozen_stages = list(frozen_stages)

        # Define stage to module mapping for flexible freezing
        # Only define composite stages here. Individual modules are handled automatically 
        # in _freeze_stages/is_frozen if they match attribute names.
        self.stage_mapping = {
            # Composite stages
            'mask_stage': [
                'mask_decoder', 'mask_head', 'query_feat', 
                'query_embed', 'level_embed', 'mask_denoiser', 'ldaf_head'
            ],
            'map_stage': [
                'map_query_generator', 'map_decoder', 'map_head', 'map_denoiser'
            ],
            'bbox_stage': ['bbox_decoder', 'bbox_head'],
        }

        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        
        # Update head configs - only set train_cfg/test_cfg if not already defined
        # This allows heads to have their own train_cfg (e.g., with assigner/sampler)
        # Update head configs and build heads
        for head_name, head_cfg in [
            ('mask_head', mask_head),
            ('map_head', map_head),
            ('map_vectorized_head', map_vectorized_head),
            ('panoptic_fusion_head', panoptic_fusion_head),
            ('bbox_head', bbox_head),
            ('ldaf_head', ldaf_head),  # NEW: LDAF Head
            ('ldaf_vectorized_head', ldaf_vectorized_head),  # NEW: LDAF Vectorized Head
        ]:
            if head_cfg is not None:
                head_cfg.setdefault('train_cfg', train_cfg)
                head_cfg.setdefault('test_cfg', test_cfg)
                setattr(self, head_name, MODELS.build(head_cfg))
            else:
                setattr(self, head_name, None)

        if ldaf_postprocessor is not None:
            self.ldaf_postprocessor = MODELS.build(ldaf_postprocessor)
        else:
            self.ldaf_postprocessor = None
        
        self._init_layers()

        self._freeze_stages()

    def _freeze_stages(self) -> None:
        """Freeze parameters and set eval mode for modules in frozen_stages."""
        if not self.frozen_stages:
            return
            
        modules_to_freeze = set()
        for stage in self.frozen_stages:
            if stage in self.stage_mapping:
                modules_to_freeze.update(self.stage_mapping[stage])
            else:
                modules_to_freeze.add(stage)
                
        for module_name in modules_to_freeze:
            module = getattr(self, module_name, None)
            if module is not None:
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
    
    def _is_frozen(self, stage_or_module: str) -> bool:
        """Check if a stage or module is frozen.
        
        Args:
            stage_or_module (str): Name of stage (e.g. 'mask_stage') 
                or module (e.g. 'backbone').
        """
        if stage_or_module in self.frozen_stages:
            return True
        
        # Check if the module is part of any frozen stage
        for frozen_stage in self.frozen_stages:
            if frozen_stage in self.stage_mapping:
                if stage_or_module in self.stage_mapping[frozen_stage]:
                    return True
        
        # Also check if stage_or_module is a stage name, and all its modules are frozen
        if stage_or_module in self.stage_mapping:
            return all(self._is_frozen(m) for m in self.stage_mapping[stage_or_module])
            
        return False

    def _get_grad_context(self, stage_or_module: str):
        """Get appropriate gradient context for a stage or module."""
        if self._is_frozen(stage_or_module):
            return torch.no_grad()
        return contextlib.nullcontext()
    
    def train(self, mode: bool = True) -> None:
        """Override train to keep frozen stages in eval mode."""
        super().train(mode)
        self._freeze_stages()
    
    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and heads."""
        # Build BEV encoder (for generating multi-scale BEV features)
        if self.bev_encoder_cfg is not None:
            self.bev_encoder = MODELS.build(self.bev_encoder_cfg)
        else:
            # Default simple BEV encoder
            self.bev_encoder = None
        
        # Build BEV neck (for multi-scale features)
        if self.bev_neck_cfg is not None:
            self.bev_neck = MODELS.build(self.bev_neck_cfg)
        else:
            self.bev_neck = None
        
        # Build mask decoder
        if self.mask_decoder_cfg is not None:
            self.mask_decoder = MODELS.build(self.mask_decoder_cfg)
            self.embed_dims = self.mask_decoder.embed_dims
        else:
            self.embed_dims = 256  # default
            self.mask_decoder = None
        
        # Build map decoder
        if self.map_decoder_cfg is not None:
            self.map_decoder = MODELS.build(self.map_decoder_cfg)
        else:
            self.map_decoder = None
        
        # Build bbox decoder
        if self.bbox_decoder_cfg is not None:
            self.bbox_decoder = MODELS.build(self.bbox_decoder_cfg)
        else:
            self.bbox_decoder = None
        
        # Build positional encoding
        if self.positional_encoding_cfg is not None:
            self.positional_encoding = MODELS.build(self.positional_encoding_cfg)
        else:
            self.positional_encoding = None
    
        # Query features (content) and query embeddings (positional)
        self.query_feat = nn.Embedding(self.num_queries, self.embed_dims)
        self.query_embed = nn.Embedding(self.num_queries, self.embed_dims)

              # Build map query generator (handles all query generation and refinement)
        if self.map_query_generator_cfg is not None:
            self.map_query_generator = TASK_UTILS.build(self.map_query_generator_cfg)
        else:
            self.map_query_generator = None
        
        # Build mask denoiser (for IMPNet stage)
        if self.mask_denoiser_cfg is not None:
            self.mask_denoiser = TASK_UTILS.build(self.mask_denoiser_cfg)
        else:
            self.mask_denoiser = None
        
        # Build map denoiser (for MMPNet stage)
        if self.map_denoiser_cfg is not None:
            self.map_denoiser = TASK_UTILS.build(self.map_denoiser_cfg)
        else:
            self.map_denoiser = None

        # Level embeddings for multi-scale features
        if self.num_transformer_feat_level > 0:
            self.level_embed = nn.Embedding(self.num_transformer_feat_level, self.embed_dims)
        
        # Get num_heads from mask decoder for attention mask generation
        if hasattr(self, 'mask_decoder') and self.mask_decoder is not None:
            if hasattr(self.mask_decoder, 'layers') and len(self.mask_decoder.layers) > 0:
                first_layer = self.mask_decoder.layers[0]
                if hasattr(first_layer, 'attentions') and len(first_layer.attentions) > 0:
                    self.num_heads = first_layer.attentions[0].num_heads
                elif hasattr(first_layer, 'cross_attn'):
                    self.num_heads = first_layer.cross_attn.num_heads
                else:
                    self.num_heads = 8  # default
            else:
                self.num_heads = 8
        else:
            self.num_heads = 8
    
    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:

        if isinstance(batch_inputs, dict):
            if 'losslessmaps' in batch_inputs:
                x = batch_inputs['losslessmaps']
            else:
                x = list(batch_inputs.values())[0]
        else:
            x = batch_inputs
        
        with self._get_grad_context('backbone'):
            x = self.backbone(x)
        
        if self.with_neck:
            with self._get_grad_context('neck'):
                x = self.neck(x)
        
        return x
    
    def pre_transformer(self,
                        mlvl_feats: Tuple[Tensor],
                        batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:
        
        encoder_inputs_dict = dict(
            mlvl_feats=mlvl_feats,
            batch_data_samples=batch_data_samples
        )

        decoder_inputs_dict = dict()

        return encoder_inputs_dict, decoder_inputs_dict

    def forward_encoder(self,
                        mlvl_feats: Tuple[Tensor],
                        batch_data_samples: OptSampleList = None,
                        **kwargs) -> Dict:
        
        if self.bev_encoder is not None:
            with self._get_grad_context('bev_encoder'):
                mlvl_feats = self.bev_encoder(mlvl_feats)
        
        with self._get_grad_context('bev_neck'):
            mask_features, memory = self.bev_neck(mlvl_feats)
        
        encoder_outputs_dict = dict(
            memory=memory,
            mask_features=mask_features,
            batch_data_samples=batch_data_samples,
        )

        return encoder_outputs_dict

    def pre_mask_decoder(self,
                         memory: Union[Tensor, List[Tensor]],
                         mask_features: Tensor,
                         batch_data_samples: OptSampleList = None,
                         **kwargs) -> Tuple[Dict, Dict]:
        
        multi_scale_memorys = memory
        if not isinstance(memory, list):
            multi_scale_memorys = [memory]
        
        batch_size = multi_scale_memorys[0].size(0)

        decoder_inputs = []
        decoder_positional_encodings = []
        multi_scale_shapes = []

        for i in range(self.num_transformer_feat_level):
            feat = multi_scale_memorys[i]
            B, C, H, W = feat.shape
            multi_scale_shapes.append((H, W))

            # Flatten and add level embedding
            decoder_input = feat.flatten(2).permute(2, 0, 1)

            if hasattr(self, 'level_embed'):
                level_emb = self.level_embed.weight[i].view(1, 1, -1)
                decoder_input = decoder_input + level_emb
            
            # Generate positional encoding
            mask = decoder_input.new_zeros((batch_size, H, W), dtype=torch.bool)
            decoder_positional_encoding = self.positional_encoding(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(2).permute(2, 0, 1)

            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)

        # Prepare query and query_pos
        query_feat = self.query_feat.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))
        
        # Apply mask denoising if training (Mask-DINO)
        dn_meta = None
        attn_mask_for_denoising = None
        if self.training and self.mask_denoiser is not None:
            # TODO: Implement mask denoising for mask stage
            # This would involve:
            # 1. Get GT masks from batch_data_samples
            # 2. Generate noised mask queries from GT
            # 3. Concatenate with original queries
            # 4. Generate attention masks
            # 5. Return dn_meta for loss computation
            #
            # Example structure (to be implemented):
            # gt_masks = self._get_gt_masks(batch_data_samples)
            # query_feat, query_embed, attn_mask_for_denoising = \
            #     self.mask_denoiser.generate(
            #         query_feat, query_embed, gt_masks, gt_labels)
            # dn_meta = self.mask_denoiser.get_dn_meta()
            pass
            
        decoder_inputs_dict = dict(
            query=query_feat,
            query_pos=query_embed,
            decoder_inputs=decoder_inputs,
            decoder_positional_encodings=decoder_positional_encodings,
            mask_features=mask_features,
            multi_scale_shapes=multi_scale_shapes)
        
        mask_head_inputs_dict = dict(
            mask_features=mask_features,
            dn_meta=dn_meta
        )

        return decoder_inputs_dict, mask_head_inputs_dict

    def forward_mask_decoder(self,
                             query: Tensor,
                             query_pos: Tensor,
                             decoder_inputs: List[Tensor],
                             decoder_positional_encodings: List[Tensor],
                             mask_features: Tensor,
                             multi_scale_shapes: List[Tuple[int, int]],
                             attn_mask: Tensor = None,
                             **kwargs) -> Dict:
        
        if self.mask_decoder is None:
            return dict(
                mask_aware_query_feat=query,
                mask_pred_list=[],
                cls_pred_list=[]
            )

        num_decoder_layers = len(self.mask_decoder.layers)

        all_query_feats = []
        cls_pred_list = []
        mask_pred_list = []

        # Generate initial attention mask
        # Use query_feat to generate initial predictions
        cls_pred, mask_pred, attn_mask = self._forward_mask_head(
            query, mask_features, multi_scale_shapes[0])
        
        all_query_feats.append(query)
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        # Iterate through decoder layers
        for i in range(num_decoder_layers):
            level_idx = i % len(decoder_inputs)

            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(
                attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            
            # Forward through decoder layer
            layer = self.mask_decoder.layers[i]
            attn_masks = [attn_mask, None]
            query = layer(
                query=query,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_pos,
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=attn_masks,
                query_key_padding_mask=None,
                key_padding_mask=None
            )

            # Generate predictions for next layer
            cls_pred, mask_pred, attn_mask = self._forward_mask_head(
                query, mask_features,
                multi_scale_shapes[(i + 1) % len(decoder_inputs)])
            
            all_query_feats.append(query)
            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
        
        mask_decoder_outputs_dict = dict(
            mask_aware_query_feat=query,
            hidden_states=torch.stack(all_query_feats, dim=0),
            cls_pred_list=cls_pred_list,
            mask_pred_list=mask_pred_list,
        )

        return mask_decoder_outputs_dict

    def _forward_mask_head(self,
                           decoder_out: Tensor,
                           mask_feature: Tensor,
                           attn_mask_target_size: Tuple[int, int]) -> Tuple[Tensor]:
        
        if hasattr(self.mask_decoder, 'post_norm'):
            decoder_out = self.mask_decoder.post_norm(decoder_out)
        decoder_out = decoder_out.transpose(0, 1)
        # shape (batch_size, num_queries, c)

        # Use panoptic_head's cls_embed and mask_embed
        cls_pred = self.mask_head.cls_embed(decoder_out)
        # shape (batch_size, num_queries, num_classes + 1)
        mask_embed = self.mask_head.mask_embed(decoder_out)
        # shape (batch_size, num_queries, c)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)

        attn_mask = F.interpolate(mask_pred,
                                  attn_mask_target_size,
                                  mode='bilinear',
                                  align_corners=False)
        # shape (batch_size, num_queries, h, w) ->
        #   (batch_size * num_heads, num_queries, h, w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return cls_pred, mask_pred, attn_mask

    def forward_transformer(self,
                            img_feats: Tuple[Tensor],
                            batch_data_samples: OptSampleList = None,
                            predict_tasks: List[str] = None) -> Dict:
        
        # Determine strict dependencies
        run_all = predict_tasks is None
        run_mask = run_all or 'mask' in predict_tasks
        run_map = run_all or 'map' in predict_tasks
        run_bbox = run_all or 'bbox' in predict_tasks
        run_ldaf = run_all or 'ldaf' in predict_tasks

        # Pre-transformer
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)
        
        # bev encoding
        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        with self._get_grad_context('mask_stage'):
            # mask decoder
            if run_mask:
                tmp_mask_dec_in, mask_head_inputs_dict = self.pre_mask_decoder(**encoder_outputs_dict)
                decoder_inputs_dict.update(tmp_mask_dec_in)

                mask_decoder_outputs_dict = self.forward_mask_decoder(**decoder_inputs_dict)
                mask_head_inputs_dict.update(mask_decoder_outputs_dict)
            else:
                mask_head_inputs_dict = None
                mask_decoder_outputs_dict = None
                
                # If only running LDAF, we still need mask_features
                if run_ldaf:
                    mask_head_inputs_dict = dict(
                        mask_features=encoder_outputs_dict['mask_features']
                    )

        if self.map_head is not None and run_map and mask_decoder_outputs_dict is not None:
            with self._get_grad_context('map_stage'):
                # Stage 4: Map decoder preparation (uses mask outputs)
                tmp_map_dec_in, map_head_inputs_dict = self.pre_map_decoder(
                    **encoder_outputs_dict,
                    **mask_decoder_outputs_dict)
        
                # Stage 5: Map decoder forward (MMPNet)
                map_decoder_outputs_dict = self.forward_map_decoder(**tmp_map_dec_in)
                map_head_inputs_dict.update(map_decoder_outputs_dict)
        else:
            map_head_inputs_dict = None

        if self.bbox_head is not None and run_bbox:
            with self._get_grad_context('bbox_stage'):
                if self.bbox_decoder:
                    x = self.bbox_decoder(encoder_inputs_dict['mlvl_feats'])
                else:
                    # 'memory' from encoder_outputs_dict contains multi-scale features.
                    x = encoder_outputs_dict['memory']
                bbox_head_inputs_dict = dict(x=x)
        else:
            bbox_head_inputs_dict = None
            
        # Combine outputs for heads
        head_inputs_dict = {
            'mask_head_inputs': mask_head_inputs_dict,
            'map_head_inputs': map_head_inputs_dict,
            'bbox_head_inputs': bbox_head_inputs_dict
        }
        
        return head_inputs_dict

    def pre_map_decoder(self,
                        memory: Union[Tensor, List[Tensor]],
                        mask_features: Tensor,
                        mask_aware_query_feat: Tensor,
                        mask_pred_list: List[Tensor],
                        batch_dada_samples: OptSampleList = None,
                        **kwargs) -> Tuple[Dict, Dict]:
        pass

    def forward_map_decoder(self,
                            query: Tensor,
                            query_pos: Tensor,
                            memory: Tensor,
                            reference_points: Tensor,
                            spatial_shapes: Tensor,
                            level_start_index: Tensor,
                            attn_masks: Tensor = None,
                            **kwargs) -> Dict:
        pass

    def loss(self,
             batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        pass

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None) -> Dict:
        
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats, batch_data_samples)
        return head_inputs_dict

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = False) -> SampleList:
        
        inference_tasks = None
        if self.test_cfg:
            inference_tasks = self.test_cfg.get('inference_tasks', None)
        
        if inference_tasks is None:
            inference_tasks = ['mask', 'map', 'bbox']
        
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(
            img_feats, batch_data_samples, predict_tasks=inference_tasks)
    
        results_list = []

        if 'mask' in inference_tasks and self.mask_head is not None and head_inputs_dict['mask_head_inputs']:
            results = self.mask_head.predict(
                **head_inputs_dict['mask_head_inputs'],
                rescale=rescale,
                batch_data_samples=batch_data_samples)

            if self.panoptic_fusion_head is not None:
                mask_cls_results, mask_pred_results = results
                results_list = self.panoptic_fusion_head.predict(
                    mask_cls_results,
                    mask_pred_results,
                    batch_data_samples,
                    rescale=rescale)
            else:
                results_list = results
            
            batch_data_samples = self.add_mask_pred_to_datasample(
                batch_data_samples, results_list)
        
        if 'map' in inference_tasks:
            
            if self.map_vectorized_head:
                # VectorizedHead generates map predictions from segmentation masks
                if head_inputs_dict['mask_head_inputs'] is not None:
                    results = self.map_vectorized_head.predict(
                        cls_pred_list=head_inputs_dict['mask_head_inputs']['cls_pred_list'],
                        mask_pred_list=head_inputs_dict['mask_head_inputs']['mask_pred_list'],
                        rescale=rescale,
                        batch_data_samples=batch_data_samples)
                    
                    batch_data_samples = self.add_map_pred_to_datasample(
                        batch_data_samples, results)
            
            elif self.map_head is not None and head_inputs_dict['map_head_inputs']:
                # Regular map head (MapTRV2Head)
                results = self.map_head.predict(
                    **head_inputs_dict['map_head_inputs'],
                    rescale=rescale,
                    batch_data_samples=batch_data_samples)
                
                batch_data_samples = self.add_map_pred_to_datasample(
                    batch_data_samples, results)

        if 'bbox' in inference_tasks and self.bbox_head is not None and head_inputs_dict['bbox_head_inputs']:
            results = self.bbox_head.predict(
                batch_data_samples=batch_data_samples,
                rescale=rescale,
                **head_inputs_dict['bbox_head_inputs'])
                
            # Add predictions to data samples
            batch_data_samples = self.add_bbox_pred_to_datasample(
                batch_data_samples, results)
        
        if 'ldaf' in inference_tasks and getattr(self, 'ldaf_head', None) is not None:
            mask_features = None
            if head_inputs_dict.get('mask_head_inputs') and 'mask_features' in head_inputs_dict['mask_head_inputs']:
                mask_features = head_inputs_dict['mask_head_inputs']['mask_features']
                
            if mask_features is not None:
                # Get target size
                if rescale and len(batch_data_samples) > 0:
                    ori_h, ori_w = batch_data_samples[0].ori_shape[:2]
                else:
                    ori_h, ori_w = batch_data_samples[0].img_shape[:2]

                # Predict LDAF fields
                dist_pred, angle_pred = self.ldaf_head.predict(
                    mask_features,
                    target_size=(ori_h, ori_w)
                )
                
                # 1. Add raw LDAF predictions to data samples
                batch_data_samples = self.add_ldaf_pred_to_datasample(
                    batch_data_samples, dist_pred, angle_pred)

                # 2. If vectorized head exists, execute automatically
                if getattr(self, 'ldaf_vectorized_head', None) is not None:
                    results = self.ldaf_vectorized_head.predict(
                        dist_pred,
                        angle_pred,
                        batch_data_samples,
                        rescale=rescale
                    )
                    
                    # Store results in 'ldaf_pred_instances'
                    for data_sample, pred_inst in zip(batch_data_samples, results):
                        data_sample.map_pred_instances = pred_inst

        return batch_data_samples

    def add_mask_pred_to_datasample(self, data_samples: SampleList,
                               results_list: List[dict]) -> SampleList:
        """Add predictions to `DetDataSample`.

        Args:
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
            results_list (List[dict]): Instance segmentation, segmantic
                segmentation and panoptic segmentation results.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances' and `pred_panoptic_seg`. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).

            And the ``pred_panoptic_seg`` contains the following key

                - sem_seg (Tensor): panoptic segmentation mask, has a
                    shape (1, h, w).
        """
        for data_sample, pred_results in zip(data_samples, results_list):
            if 'pan_results' in pred_results:
                data_sample.pred_panoptic_seg = pred_results['pan_results']

            if 'ins_results' in pred_results:
                data_sample.mask_pred_instances = pred_results['ins_results']

            assert 'sem_results' not in pred_results, 'segmantic ' \
                'segmentation results are not supported yet.'

        return data_samples

    def add_bbox_pred_to_datasample(self, data_samples: SampleList,
                                   results_list) -> SampleList:
        """Add bbox predictions to `DetDataSample`.

        Args:
            data_samples (list[:obj:`DetDataSample`]): Batch of data samples.
            results_list (List[InstanceData]): Detection results.

        Returns:
            list[:obj:`DetDataSample`]: Updated data samples.
        """
        data_samples = self.add_pred_to_datasample(data_samples, results_list)

        return data_samples

    def add_ldaf_pred_to_datasample(self, data_samples: SampleList,
                                   dist_pred: Tensor,
                                   angle_pred: Tensor) -> SampleList:
        """Add LDAF predictions to `DetDataSample`.
        
        Args:
            data_samples (list[:obj:`DetDataSample`]): Batch of data samples.
            dist_pred (Tensor): Distance predictions (B, C, H, W).
            angle_pred (Tensor): Angle predictions (B, C, 2, H, W).
            
        Returns:
            list[:obj:`DetDataSample`]: Updated data samples.
        """
        from xdlt.dl.framework.structures import InstanceData
        
        # Move to CPU for postprocessing (skeletonization is CPU-bound usually)
        dist_pred_cpu = dist_pred.detach().cpu().numpy()
        angle_pred_cpu = angle_pred.detach().cpu().numpy()
        
        for i, data_sample in enumerate(data_samples):
            # 1. Store Raw LDAF
            ldaf_results = dict(
                distance=dist_pred[i],
                angle=angle_pred[i]
            )
            setattr(data_sample, 'pred_ldaf', ldaf_results)
            
            # 2. Run PostProcessor if available
            if self.ldaf_postprocessor is not None:
                # Per-class extraction
                num_classes = dist_pred.shape[1]
                all_lines = []
                all_labels = []
                all_scores = [] # LDAF doesn't give instance scores, assume 1.0 or mean dist?
                
                # Check if we should upscale for post-processing?
                # The dist/angle fields might be low-res. 
                # extract_lines expects fields.
                
                for c in range(num_classes):
                    d_map = dist_pred_cpu[i, c]
                    a_map = angle_pred_cpu[i, c]
                    
                    # Extract
                    # input fields are (H, W) and (2, H, W)
                    lines = self.ldaf_postprocessor.extract_lines(d_map, a_map)
                    
                    # Scale lines to input size (Assuming 4x downsample)
                    stride = 4.0
                    scaled_lines = []
                    for line in lines:
                        scaled_lines.append(line * stride)
                    lines = scaled_lines
                    
                    # DEBUG
                    # if len(lines) > 0:
                    #     print(f"[Mask2Map] Class {c}: Extracted {len(lines)} lines")
                    
                    for line in lines:
                        all_lines.append(line)
                        all_labels.append(c)
                        all_scores.append(1.0) # Placeholder score
                        
                # Pack into InstanceData
                inst = InstanceData()
                inst.lines = all_lines # List[np.ndarray]
                inst.labels = torch.tensor(all_labels, dtype=torch.long)
                inst.scores = torch.tensor(all_scores, dtype=torch.float32)
                
                # Also assign to vectors for compatibility with LineMetric
                # inst.vectors = all_lines 
                
                # Assign to data_sample
                # Note: Mask2Map usually puts bbox/mask in 'pred_instances'.
                # We can merge or just overwrite if only ldaf task.
                # If map task also exists, 'map_pred_instances' is used.
                
                # Let's verify where LineMetric looks: pred_instances.lines
                # So we put it in pred_instances.
                
                if hasattr(data_sample, 'pred_instances'):
                    # Merge if exists? Or just append fields
                    data_sample.pred_instances.lines = all_lines
                    # If empty
                    if len(data_sample.pred_instances) == 0:
                        data_sample.pred_instances = inst
                else:
                    data_sample.pred_instances = inst
                    
        return data_samples

    def add_map_pred_to_datasample(self, data_samples, results_list):
        """Add predictions to `DetDataSample`.
        
        MapTRV2 uses a different prediction format than standard detectors.
        Each result is a dict with instance types as keys (e.g., 'laneboundary', 'roadedge').
        
        Args:
            data_samples (list[:obj:`DetDataSample`]): Batch of data samples.
            results_list (list[dict]): Prediction results. Each element is a dict
                with instance type names as keys, and each value contains:
                - scores (Tensor): Classification scores
                - labels (Tensor): Labels
                - instance_points (Tensor): Predicted polyline points
                - frame_root (str): Frame identifier
        
        Returns:
            list[:obj:`DetDataSample`]: Updated data samples with predictions.
        """
        from xdlt.dl.framework.structures import InstanceData
        
        for data_sample, pred_results in zip(data_samples, results_list):
            # pred_results is a dict like:
            # {
            #   'laneboundary': {'scores': ..., 'labels': ..., 'instance_points': ...},
            #   'roadedge': {...},
            #   ...
            # }
            
            # Create a combined InstanceData for all instance types
            all_scores = []
            all_labels = []
            all_points = []
            all_types = []  # Track which instance type each prediction belongs to
            
            for inst_type, inst_result in pred_results.items():
                scores = inst_result['scores']
                labels = inst_result['labels']
                points = inst_result['instance_points']
                
                if len(scores) > 0:
                    all_scores.append(scores)
                    all_labels.append(labels)
                    all_points.append(points)
                    # Create type indicator tensor
                    type_tensor = torch.full((len(scores),), 
                                            hash(inst_type) % 1000,  # Simple hash for type ID
                                            dtype=torch.long,
                                            device=scores.device if torch.is_tensor(scores) else torch.device('cpu'))
                    all_types.append(type_tensor)
            
            # Combine all predictions
            pred_instances = InstanceData()
            if all_scores:
                pred_instances.scores = torch.cat([s if torch.is_tensor(s) else torch.tensor(s) 
                                                   for s in all_scores])
                pred_instances.labels = torch.cat([l if torch.is_tensor(l) else torch.tensor(l) 
                                                   for l in all_labels])
                pred_instances.vectors = torch.cat([p if torch.is_tensor(p) else torch.tensor(p) 
                                                           for p in all_points])
                pred_instances.types = torch.cat(all_types)
            else:
                # Empty predictions
                pred_instances.scores = torch.tensor([])
                pred_instances.labels = torch.tensor([])
                pred_instances.vectors = torch.tensor([])
                pred_instances.types = torch.tensor([])
            
            # # Store the original dict format as well for compatibility
            # pred_instances.pred_results_dict = pred_results
            
            data_sample.map_pred_instances = pred_instances
        
        return data_samples