FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

ARG TORCH_CUDA_ARCH_LIST="12.0"
ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Replace Ubuntu sources with Tsinghua mirror (China)
RUN sed -i 's@archive.ubuntu.com@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list \
    && sed -i 's@security.ubuntu.com@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list

# Install system dependencies needed by OpenMMLab source builds.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
        ninja-build \
        vim \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Configure pip to use Tsinghua mirror (China).
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install -U pip openmim --no-cache-dir

COPY requirements/base.txt /tmp/requirements/base.txt
RUN pip install --no-cache-dir -r /tmp/requirements/base.txt

RUN pip install --no-cache-dir "mmengine==0.10.7"

# Build MMCV 2.1.0 from source so CUDA ops are compiled for Blackwell (sm_120).
WORKDIR /opt/openmmlab
RUN wget https://github.com/open-mmlab/mmcv/archive/refs/tags/v2.1.0.tar.gz -O mmcv-2.1.0.tar.gz \
    && tar -zxf mmcv-2.1.0.tar.gz \
    && cd mmcv-2.1.0 \
    && FORCE_CUDA=1 MMCV_WITH_OPS=1 TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
        pip install -v . --no-build-isolation --no-cache-dir \
    && cd /opt/openmmlab \
    && rm -rf mmcv-2.1.0 mmcv-2.1.0.tar.gz

RUN pip install --no-cache-dir "mmdet==3.3.0"

# Install MMDetection3D 1.4.0 from source.
WORKDIR /opt/openmmlab
RUN git clone --branch v1.4.0 --depth 1 https://github.com/open-mmlab/mmdetection3d.git \
    && cd mmdetection3d \
    && pip install -v . --no-build-isolation --no-deps --no-cache-dir \
    && cd /opt/openmmlab \
    && rm -rf mmdetection3d

COPY requirements/dataset.txt /tmp/requirements/dataset.txt
RUN pip install --no-cache-dir -r /tmp/requirements/dataset.txt

COPY requirements/optional.txt /tmp/requirements/optional.txt
RUN pip install --no-cache-dir -r /tmp/requirements/optional.txt

RUN python - <<'PY'
import mmcv
import mmdet
import mmengine
import mmdet3d

expected = {
    'mmengine': ('0.10.7', mmengine.__version__),
    'mmcv': ('2.1.0', mmcv.__version__),
    'mmdet': ('3.3.0', mmdet.__version__),
    'mmdet3d': ('1.4.0', mmdet3d.__version__),
}
for name, (want, got) in expected.items():
    if got != want:
        raise RuntimeError(f'{name} version mismatch: expected {want}, got {got}')
print('OpenMMLab versions OK:', expected)
PY

WORKDIR /workspace/mmhdmap

CMD ["bash"]
