# C-Radio-Path-Exp training environment (Route A, containerized).
#
# Reproduces the verified training environment in one build:
#   python_version.txt (3.10.20) + pip_freeze_tao_v7_py310.txt (180 pins,
#   torch==2.11.0+cu126). The CUDA-12.6 torch wheels bundle their own
#   CUDA/NCCL runtime, so the image needs only the NVIDIA driver on the
#   host — run with --gpus all.
#
# Build:  docker build -t c-radio-path-exp environment/
# Smoke:  docker run --gpus all --rm -it \
#             -v /path/to/C-Radio-Path-Exp:/kit -w /kit/run \
#             -e NUM_GPUS=1 -e MAX_STEPS=10 c-radio-path-exp ./start_training.sh
# Train:  same command without NUM_GPUS/MAX_STEPS (defaults to 8 GPUs).

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Runtime libs needed by the OpenCV/Pillow wheels in the freeze
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
 && rm -rf /var/lib/apt/lists/*

# 1) exact torch/torchvision builds from the CUDA-12.6 wheel index
# 2) every remaining pin from the verified freeze
COPY pip_freeze_tao_v7_py310.txt /tmp/pip_freeze.txt
RUN pip install torch==2.11.0 torchvision==0.26.0 \
      --index-url https://download.pytorch.org/whl/cu126 \
 && grep -vE '^(torch|torchvision)==' /tmp/pip_freeze.txt > /tmp/pins.txt \
 && pip install -r /tmp/pins.txt \
 && rm -f /tmp/pip_freeze.txt /tmp/pins.txt \
 && python -c "import torch, pytorch_lightning, timm, webdataset, hydra; print('env OK:', torch.__version__)"

WORKDIR /kit/run
CMD ["/bin/bash"]
