from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import Registry

# manage all kinds of modules inheriting `nn.Module`
MODELS = Registry('model', parent=MMENGINE_MODELS, locations=['mmhdmap.models'])

ACTIVATION = Registry('activation', scope='mmhdmap', locations=['mmhdmap.models.activations'])

DATASETS = Registry('dataset', parent=MMENGINE_DATASETS, locations=['mmhdmap.datasets'])

METRICS = Registry('metric', parent=MMENGINE_METRICS, locations=['mmhdmap.metrics'])

TRANSFORMS = Registry('transform', parent=MMENGINE_TRANSFORMS, locations=['mmhdmap.transforms'])