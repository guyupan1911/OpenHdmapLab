FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

ARG http_proxy
ARG https_proxy
ARG no_proxy

ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV no_proxy=${no_proxy}

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Replace Ubuntu sources with Tsinghua mirror (China)
RUN sed -i 's@archive.ubuntu.com@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list && \
    sed -i 's@security.ubuntu.com@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Configure pip to use Tsinghua mirror (China)
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install OpenMMLab package manager
RUN pip install -U openmim --no-cache-dir

# Install OpenMMLab packages
RUN mim install mmengine
RUN mim install "mmcv==2.1.0"
RUN mim install "mmdet==3.3.0"

RUN pip install -U numpy==1.26.4

# TODO: Add more OpenMMLab packages if needed
# Example:
# RUN mim install mmsegmentation
# RUN mim install mmpose

CMD ["bash"]
