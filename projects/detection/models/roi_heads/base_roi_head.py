from abc import ABCMeta, abstractmethod
from typing import Tuple

from mmengine.model import BaseModule
from torch import Tensor

from mmdet.structures import SampleList
from mmdet.utils import InstanceList, OptConfigType, OptMultiConfig

from mmhdmap.registry import MODELS


class BaseRoIHead(BaseModule, metaclass=ABCMeta):

    def __init__(self,
                 bbox_roi_extractor: OptMultiConfig = None,
                 bbox_head: OptMultiConfig = None,
                 mask_roi_extractor: OptMultiConfig = None,
                 mask_head: OptMultiConfig = None,
                 shared_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        
        super().__init__(init_cfg=init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if shared_head is not None:
            self.shared_head = MODELS.build(shared_head)
        
        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, bbox_head)
        
        if mask_head is not None:
            self.init_mask_head(mask_roi_extractor, mask_head)
        
        self.init_assigner_sampler()
    
    @property
    def with_bbox(self) -> bool:
        """bool: whether the RoI head contains a `bbox_head`"""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self) -> bool:
        """bool: whether the RoI head contains a `mask_head`"""
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @property
    def with_shared_head(self) -> bool:
        """bool: whether the RoI head contains a `shared_head`"""
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @abstractmethod
    def init_bbox_head(self, *args, **kwargs):
        """Initialize ``bbox_head``"""
        pass

    @abstractmethod
    def init_mask_head(self, *args, **kwargs):
        """Initialize ``mask_head``"""
        pass

    @abstractmethod
    def init_assigner_sampler(self, *args, **kwargs):
        """Initialize assigner and sampler."""
        pass
    
    @abstractmethod
    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: SampleList):
        pass
    
    def predict(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        
        assert self.with_bbox, 'Bbox head must be implemented.'
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples]
        
        bbox_rescale = rescale if not self.with_mask else False
        results_list = self.predict_bbox(
            x,
            batch_img_metas,
            rpn_results_list,
            rcnn_test_cfg=self.test_cfg,
            rescale=bbox_rescale)
        
        if self.with_mask:
            results_list = self.predict_mask(
                x, batch_img_metas, results_list, rescale=rescale)
        
        return results_list