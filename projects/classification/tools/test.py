from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmengine.runner import Runner
from mmengine.registry import MODELS, DATASETS


cfg = Config.fromfile('projects/classification/configs/resnet50.py')

runner = Runner.from_cfg(cfg)
runner.train()


