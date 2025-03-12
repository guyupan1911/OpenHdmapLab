from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import Registry

# manage all kinds of modules inheriting `nn.Module`
MODELS = Registry('model', parent=MMENGINE_MODELS, locations=['mmhdmap.models'])