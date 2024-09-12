#
# Dockerfile for 
# StraightPCF: Straight Point Cloud Filtering (CVPR 2024)
# https://github.com/ddsediri/StraightPCF
#

ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=11.8.0

FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-cudnn8-devel-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND noninteractive
WORKDIR /app/

# APT packages
RUN apt-get update && apt-get install --no-install-recommends -y \
    nano \
    git \
    locales \
    ca-certificates \
    wget \
    g++ \
    cmake \
    python3 \
    python3-dev \
    python3-distutils \
    python3-pip \
    build-essential \
    libeigen3-dev \
    libcgal-dev \
    libgflags-dev \
    libatlas-base-dev

# PointFilter deps
RUN pip install --upgrade setuptools && pip install --upgrade pip
RUN pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install numpy matplotlib scipy scikit-learn plyfile pandas tensorboard torchsummary fvcore iopath torch-cluster point-cloud-utils \
    && FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# PointFilter
RUN git clone https://github.com/alancneves/StraightPCF StraightPCF
RUN ln -s $(which python3) /usr/bin/python