# OpenHdmapLab

OpenHdmapLab is a deep learning project for HD map scene understanding and element extraction. It integrates a Mask2Former-based instance/panoptic segmentation and map-element prediction pipeline, and provides inference and visualization tools to generate semantic map outputs such as lane markings, curbs, and crosswalks from images.

## Model

The core model is `Mask2Map`, built on a Swin Transformer backbone with a multi-scale deformable attention pixel decoder. It uses a transformer decoder with learned queries to produce instance/panoptic masks, and includes an LDAF head for learning distance-and-angle fields to refine thin-structure map elements. Inference outputs are visualized with a custom RotLocalVisualizer and color palette tailored to HD map classes.

## Install

```bash
# 1) Create an environment
conda create -n openmmlab python=3.10 -y
conda activate openmmlab

# 2) Install PyTorch (CUDA 12.x)
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3) Install OpenMMLab dependencies
pip install -U openmim
mim install "mmcv>=2.0.0"
pip install mmengine mmdet

# 4) Other dependencies
pip install opencv-python

# 5) Install this project
pip install -e .
```

## Inference Example
```bash
cd mmhdmap
python projects/mask2former/tools/inference_single.py \
  --config projects/mask2former/configs/mask2former_swim_ldaf.py \
  --checkpoint data/mask2map_ckpts/rbbox.pth \
  --image_path projects/mask2former/tools/LosslessMap.jpg \
  --output_dir projects/mask2former/tools/
```

## Segmentation Result
![segmentation_result](assets/segmentation_preview.png)

test push 1
