# Base image for building the environment
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python package manager
RUN apt-get update && apt-get install -y python3-dev python3-pip libvips

# Install PyTorch and torchvision with specific versions and cu117 support
RUN pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# Install additional libraries
RUN pip install tqdm addict easydl einops h5py numpy==1.26.4 pandas Pillow pyarrow pyparsing PyYAML scanpy scikit-learn scipy scprep tifffile==2022.10.10 torchmetrics==0.10.2 pytorch-lightning==1.8.0.post1 pyvips

# Copy your application code (if applicable)

CMD [ "/bin/bash" ]  # Replace with your application script
