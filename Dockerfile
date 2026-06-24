# Unified-Lift training environment (CUDA 11.3 / PyTorch 1.12.1)
# Built for NVIDIA RTX A6000 (sm_86).
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
# A6000 = compute capability 8.6; set so the CUDA submodules compile the right arch.
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 python3.8-dev python3-pip git ninja-build \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.8 /usr/bin/python

RUN python -m pip install --no-cache-dir --upgrade "pip<24" setuptools wheel

# PyTorch 1.12.1 + cu113 (matches the CUDA 11.3 base toolkit)
RUN pip install --no-cache-dir \
        torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
        --extra-index-url https://download.pytorch.org/whl/cu113

# Python deps from doc/Usage.md (numpy pinned for torch 1.12 / sklearn compat)
RUN pip install --no-cache-dir \
        numpy==1.23.5 plyfile==0.8.1 tqdm scipy wandb \
        opencv-python-headless scikit-learn lpips

# Build the two CUDA extensions into site-packages so they persist in the image.
COPY submodules /tmp/submodules
RUN pip install --no-cache-dir /tmp/submodules/diff-gaussian-rasterization \
    && pip install --no-cache-dir /tmp/submodules/simple-knn \
    && rm -rf /tmp/submodules

WORKDIR /workspace
