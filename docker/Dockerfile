FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

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
RUN pip install -U "numpy==1.26.4" "python-lzf==0.2.6"
RUN mim install "mmengine==0.10.7"
RUN mim install "mmcv==2.1.0"
RUN mim install "mmdet==3.3.0"
RUN mim install "mmdet3d==1.4.0"

# TODO: Add more OpenMMLab packages if needed
# Example:
# RUN mim install mmsegmentation
# RUN mim install mmpose

CMD ["bash"]
