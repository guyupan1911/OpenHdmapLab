from typing import List, Optional

from mmengine.structures import BaseDataElement, InstanceData, PixelData

class DetDataSample(BaseDataElement):
    pass

SampleList = List[DetDataSample]
OptSampleList = Optional[SampleList]