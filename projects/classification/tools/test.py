from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmengine.runner import Runner
from mmengine.registry import MODELS, DATASETS


cfg = Config.fromfile('projects/classification/configs/resnet/resnet50.py')

cfg.work_dir = './work_dirs'

runner = Runner.from_cfg(cfg)
runner.train()


