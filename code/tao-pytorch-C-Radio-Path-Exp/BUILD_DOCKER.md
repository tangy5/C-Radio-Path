# TAO PyTorch Docker Build Guide

This guide explains how to build the TAO PyTorch Docker container from source for C-RADIOv4 distillation training.

## Prerequisites

### System Requirements
- Docker >= 19.03.5 with buildx support
- nvidia-docker2 >= 2.5.0-1
- nvidia-container-toolkit >= 1.3.0-1
- NVIDIA Driver >= 535.85
- At least 50GB free disk space (base image + build artifacts)
- 16GB+ RAM recommended

### Docker Login
You need to log in to NVIDIA's container registry:
```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <your NGC API key>
```

## Build Steps

### Option 1: Quick Build (Recommended for Single Platform)

```bash
# 1. Navigate to the repository
cd <CLUSTER_WORKSPACE>/tao-pytorch-custom-build

# 2. Source the environment setup
source scripts/envsetup.sh

# 3. Build for x86_64 (most common)
cd docker
./build.sh --build --x86

# 4. The image will be tagged as: nvcr.io/nvstaging/tao/tao_pytorch_base_image:$USER
```

### Option 2: Multi-Platform Build (x86 + ARM)

```bash
cd <CLUSTER_WORKSPACE>/tao-pytorch-custom-build
cd docker
./build.sh --build --multiplatform --push
```

**Note:** Multi-platform builds require pushing to registry (cannot load locally).

### Option 3: Force Rebuild (No Cache)

```bash
cd docker
./build.sh --build --force --x86
```

## Build Configuration

### Environment Variables
- `NV_TAO_PYTORCH_TOP`: Auto-set by envsetup.sh to the repo root

### Build Arguments (in Dockerfile)
- `PYTORCH_BASE_IMAGE`: Default `nvcr.io/nvidia/pytorch:25.09-py3`
- `ONNXRUNTIME_VERSION`: Default `1.23.0`
- `XFORMERS_COMMIT_HASH`: Specific commit for xformers build

## Build Outputs

After successful build:
- **Image Name**: `nvcr.io/nvstaging/tao/tao_pytorch_base_image:$USER`
- **Location**: Local Docker daemon
- **Size**: ~15-20 GB

## Using the Built Container

### Method 1: Using tao_pt CLI
```bash
# Set up environment
source <CLUSTER_WORKSPACE>/tao-pytorch-custom-build/scripts/envsetup.sh

# Run interactive container
tao_pt --gpus all \
       --volume <CLUSTER_STORAGE>:<CLUSTER_STORAGE> \
       --volume /path/to/results:/workspace/results \
       --shm_size 16G
```

### Method 2: Direct Docker Run
```bash
docker run --gpus all \
    -it \
    --rm \
    -v <CLUSTER_STORAGE>:<CLUSTER_STORAGE> \
    -v /path/to/results:/workspace/results \
    -e PYTHONPATH=/tao-pt \
    --shm-size=16g \
    nvcr.io/nvstaging/tao/tao_pytorch_base_image:$USER
```

### Method 3: SLURM with Enroot/Pyxis (SQSH)

Convert Docker to SQSH for cluster use:
```bash
# On a node with Docker
enroot import dockerd://nvcr.io/nvstaging/tao/tao_pytorch_base_image:$USER

# This creates a .sqsh file that can be used with SLURM
```

## Troubleshooting

### Issue: Build fails with "docker not found"
**Solution**: Ensure Docker is installed and you have permissions:
```bash
sudo usermod -a -G docker $(whoami)
# Log out and log back in
```

### Issue: xformers clone fails
**Solution**: The build script clones from internal NVIDIA GitLab. If you're outside NVIDIA network:
1. Comment out xformers build in Dockerfile
2. Install xformers from pip instead

### Issue: Out of disk space
**Solution**: Clean up old images:
```bash
docker system prune -a
docker volume prune
```

### Issue: CUDA/GPU not available in container
**Solution**: Ensure nvidia-docker2 is installed:
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## For C-RADIOv4 Distillation

Once the container is built, create a mounts file at `~/.tao_mounts.json`:
```json
{
    "Mounts": [
        {
            "source": "<CLUSTER_WORKSPACE>/teacher_models",
            "destination": "/workspace/teacher_models"
        },
        {
            "source": "<CLUSTER_WORKSPACE>/tao-pytorch-custom-build",
            "destination": "/tao-pt"
        },
        {
            "source": "/path/to/datasets",
            "destination": "/workspace/datasets"
        },
        {
            "source": "/path/to/experiments",
            "destination": "/workspace/experiments"
        }
    ]
}
```

Run training:
```bash
tao_pt --gpus all --env PYTHONPATH=/tao-pt

# Inside container
cd /tao-pt/nvidia_tao_pytorch/cv/classification_pyt/scripts
python distill.py --config-path ../experiment_specs --config-name debug3
```

## Next Steps

1. Build the Docker image on a node with Docker access
2. Convert to SQSH format if using SLURM with Enroot/Pyxis
3. Create your mounts configuration
4. Run the C-RADIOv4 distillation training
