import torch.nn as nn
from torch import Tensor

from mmcv.cnn import Linear
from mmcv.bricks.transformer import FFN
from mmdet.registry import MODELS


@MODELS.register_module()
class MyDetrHead(BaseModule):

    def __init__(self,
                 num_classes: int,
                 embed_dims: int = 256,
                 num_reg_fcs: int = 2,
                 test_cfg: ConfigType = dict(max_per_img=100),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg)

        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_reg_fcs = num_reg_fcs
        
        self.test_cfg = test_cfg

        self.cls_out_channels = num_classes

        self._init_layers()
    
    def _init_layers(self):
        # cls branch
        self.fc_cls = nn.Linear(self.embed_dims, self.cls_out_channels)
        # reg branch
        self.activate = nn.ReLU()
        self.reg_ffn = FFN(
            self.embed_dims,
            self.embed_dims,
            self.num_fcs,
            dict(type='ReLU', inplace=True),
            ffn_drop=0.0,
            add_identify=False)
        self.fc_reg = Linear(self.embed_dims, 4)
    
    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor]:
        # hidden_states: (decoder_layers, bs, num_queries, embed_dims)

        # -> (decoder_layers, bs, num_queries, cls_out_channels)
        layers_cls_scores = self.fc_cls(hidden_states)

        layers_bbox_preds = self.fc_reg(
            self.activate(self.reg_ffn(hidden_states))).sigmoid()
        
        return layers_cls_scores, layers_bbox_preds

    def predict(self,
                hidden_states: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:
        
        batch_img_metas = [
           data_sample.metainfo for data_sample in batch_data_samples
        ]
    
        last_layer_hidden_states = hidden_states[-1].unsqueeze()

        outs = self(last_layer_hidden_states)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
        
        return predictions

    def predict_by_feat(self,
                        layer_cls_scores: Tensor,
                        layer_bbox_preds: Tensor,
                        batch_img_metas: List[dict],
                        rescale: bool = True) -> InstanceList:
            
            batch_cls_scores = layer_cls_scores[-1]
            batch_bbox_preds = layer_bbox_preds[-1]

            results_list = []
            for img_id in range(len(batch_img_metas)):
                cls_scores = batch_cls_scores[img_id]
                bbox_preds = batch_bbox_preds[img_id]
                img_meta = batch_img_metas[img_id]
                results = self._predict_by_feat_single(
                    cls_scores, bbox_preds, img_meta, rescale)
                
                results_list.append(results)
            
            return results_list

    def _predict_by_feat_single(self,
                                cls_scores: Tensor,
                                bbox_preds: Tensor,
                                img_meta: dict,
                                rescale: bool = True) -> InstanceData:
        # cls_scores: (num_queries, cls_out_channels)
        # bbox_preds: (num_queries, 4)

        assert len(cls_scores) == len(bbox_preds)
        max_per_img = self.test_cfg.get('max_per_img', len(cls_scores))
        img_shape = img_meta['img_shape']
        
        if self.loss_cls.use_sigmoid:
            cls_scores = cls_scores.sigmoid()
            scores, indexes = cls_scores.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_indexes = indexes // self.num_classes
            det_bboxes = bbox_preds[bbox_indexes]
        else:
            # cls_scores: (num_queries, cls_out_channels)
            # -> (num_queries, 1)
            cls_scores, det_labels = F.softmax(cls_scores, -1)[..., :-1].max(-1)
            scores, indexes = cls_scores.topk(max_per_img)
            det_labels = det_labels[indexes]
            det_bboxes = bbox_preds[indexes]

        det_bboxes = bbox_cxcywh_to_xyxy(det_bboxes)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])

        if rescale:
            assert img_meta.get('scale_factor') is not None
            det_bboxes /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1,2))

        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels
        return results