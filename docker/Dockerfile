# Base image for building the environment
FROM ubuntu:24.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies for Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    libvips \
    libvips-dev \
    openslide-tools \
    git \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.11 get-pip.py \
    && rm get-pip.py

# Set working directory
WORKDIR /app

# Install PyTorch and torchvision with specific versions and cu121 support
RUN python3.11 -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# Install CUDA-related Python packages
RUN python3.11 -m pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12==24.6.0 \
    dask-cudf-cu12==24.6.0 \
    cucim-cu12==24.6.0 \
    raft-dask-cu12==24.6.0

# Clone and install HEST
RUN git clone https://github.com/mahmoodlab/HEST.git && \
    cd HEST && \
    python3.11 -m pip install -e .

# Install additional libraries
RUN python3.11 -m pip install \
    addict \
    openslide-bin \
    wget \
    wandb

RUN python3.11 -m pip install flash-attn --no-build-isolation

CMD [ "/bin/bash" ] 