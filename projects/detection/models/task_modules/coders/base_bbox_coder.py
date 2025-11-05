from abc import ABCMeta abstractmethod


class BaseBBoxCoder(metaclass=ABCMeta):
    encode_size = 4

    def __init__(self, use_box_type: bool = False, **kwargs):
        self.use_box_type = use_box_type
    
    @abstractmethod
    def encode(self, bboxes, gt_bboxes):
    
    @abstractmethod
    def decode(self, bboxes, bboxes_pred):
        