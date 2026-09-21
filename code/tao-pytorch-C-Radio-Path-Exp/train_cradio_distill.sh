#!/bin/bash
#SBATCH --account=<your_account>          # Replace with your account
#SBATCH --partition=batch                 # Or interactive
#SBATCH --job-name=cradio_distill
#SBATCH --time=24:00:00                   # Adjust based on your needs
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8                 # Use all 8 GPUs
#SBATCH --cpus-per-gpu=4                  # 32 CPUs total
#SBATCH --mem=256G                        # System memory
#SBATCH --gres=gpu:8

# C-RADIOv4 Multi-Teacher Distillation Training Script
# Based on Yu Wang's workflow

set -euo pipefail

echo "=========================================="
echo "C-RADIOv4 Distillation Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=========================================="

# Configuration - MODIFY THESE PATHS
TAO_REPO="<CLUSTER_WORKSPACE>/tao-pytorch-custom-build"
TEACHER_MODELS="<CLUSTER_WORKSPACE>/teacher_models"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-<CLUSTER_WORKSPACE>/experiments}"
DATASET_ROOT="${DATASET_ROOT:-/path/to/imagenet}"  # Update this
CONFIG_NAME="${CONFIG_NAME:-debug3}"

# Container image - use Yu Wang's or your own
CONTAINER_IMAGE="${CONTAINER_IMAGE:-<EDGEAI_ROOT>/users/<user>/docker/tao_2512.sqsh}"
# Or use your built image: <CLUSTER_WORKSPACE>/tao_pytorch_distill.sqsh

# GPU settings
export TAO_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Use all GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$TAO_REPO:$PYTHONPATH"

# Create experiment directory
mkdir -p "$EXPERIMENTS_DIR"

echo ""
echo "Configuration:"
echo "  TAO Repo: $TAO_REPO"
echo "  Teacher Models: $TEACHER_MODELS"
echo "  Experiments: $EXPERIMENTS_DIR"
echo "  Dataset: $DATASET_ROOT"
echo "  Config: $CONFIG_NAME"
echo "  Container: $CONTAINER_IMAGE"
echo ""

# Check if using Enroot/Pyxis (SQSH) or direct Docker
if [ -f "$CONTAINER_IMAGE" ]; then
    echo "Using Enroot/Pyxis with SQSH container..."
    
    # Set up container mounts
    export CONTAINER_MOUNTS="$TAO_REPO:/tao-pt,$TEACHER_MODELS:/workspace/teacher_models,$DATASET_ROOT:/workspace/datasets,$EXPERIMENTS_DIR:/workspace/experiments,$HOME:/home"
    
    echo "Container mounts: $CONTAINER_MOUNTS"
    
    # Run training inside container
    srun --container-image "$CONTAINER_IMAGE" \
         --container-mounts="$CONTAINER_MOUNTS" \
         --container-workdir=/tao-pt \
         bash -c "
            set -e
            echo 'Inside container:'
            echo '  Hostname: \$(hostname)'
            echo '  Python: \$(which python)'
            echo '  GPUs: \$(nvidia-smi -L | wc -l) GPUs available'
            echo ''
            
            # Set environment inside container
            export PYTHONPATH=/tao-pt:\$PYTHONPATH
            export TAO_VISIBLE_DEVICES=$TAO_VISIBLE_DEVICES
            
            # Navigate to scripts directory
            cd /tao-pt/nvidia_tao_pytorch/cv/classification_pyt/scripts
            
            echo 'Starting training...'
            echo 'Command: python distill.py --config-path ../experiment_specs --config-name $CONFIG_NAME'
            echo ''
            
            # Run training
            python distill.py \
                --config-path ../experiment_specs \
                --config-name $CONFIG_NAME
         "

else
    echo "ERROR: Container image not found: $CONTAINER_IMAGE"
    echo ""
    echo "Options:"
    echo "1. Build your own container (see BUILD_DOCKER.md)"
    echo "2. Use Yu Wang's container if accessible"
    echo "3. Run without container (if dependencies are installed)"
    exit 1
fi

echo ""
echo "=========================================="
echo "Training completed!"
echo "Ended: $(date)"
echo "=========================================="
