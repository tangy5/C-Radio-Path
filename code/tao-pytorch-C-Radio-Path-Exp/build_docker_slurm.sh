#!/bin/bash
#SBATCH --account=<your_account>      # Replace with your SLURM account
#SBATCH --partition=batch             # Or interactive partition
#SBATCH --job-name=tao_docker_build
#SBATCH --time=04:00:00               # Build takes 1-3 hours
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16            # Parallel build
#SBATCH --mem=64G                     # Build needs memory
#SBATCH --gres=gpu:0                  # No GPU needed for build

# TAO PyTorch Docker Build Script for SLURM
# Run this on a node with Docker access

set -euo pipefail

echo "=========================================="
echo "TAO PyTorch Docker Build Script"
echo "Started: $(date)"
echo "=========================================="

# Configuration
TAO_REPO="<CLUSTER_WORKSPACE>/tao-pytorch-custom-build"
BUILD_PLATFORM="${BUILD_PLATFORM:-linux/amd64}"  # or linux/arm64
OUTPUT_SQSH="${OUTPUT_SQSH:-<CLUSTER_WORKSPACE>/tao_pytorch_distill.sqsh}"

# Check if running in SLURM
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "Running under SLURM Job ID: $SLURM_JOB_ID"
fi

# Check Docker availability
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found. This script requires Docker."
    echo "Please run on a node with Docker installed."
    exit 1
fi

# Check Docker is working
docker info > /dev/null 2>&1 || {
    echo "ERROR: Docker daemon not accessible."
    echo "You may need to add yourself to the docker group:"
    echo "  sudo usermod -a -G docker \$(whoami)"
    exit 1
}

# Navigate to TAO repository
cd "$TAO_REPO" || {
    echo "ERROR: Cannot find TAO repository at $TAO_REPO"
    exit 1
}

echo "Repository location: $(pwd)"

# Source environment setup
echo ""
echo "Step 1: Setting up environment..."
source scripts/envsetup.sh

# Verify NV_TAO_PYTORCH_TOP is set
if [ -z "${NV_TAO_PYTORCH_TOP:-}" ]; then
    echo "ERROR: NV_TAO_PYTORCH_TOP not set"
    exit 1
fi

echo "NV_TAO_PYTORCH_TOP: $NV_TAO_PYTORCH_TOP"

# Navigate to docker directory
cd "$NV_TAO_PYTORCH_TOP/docker" || exit 1

echo ""
echo "Step 2: Building Docker image..."
echo "Platform: $BUILD_PLATFORM"
echo "This may take 1-3 hours depending on network and CPU..."
echo ""

# Build the Docker image
if [ -x "./build.sh" ]; then
    # Use the official build script
    ./build.sh --build --x86 2>&1 | tee "docker_build_$(date +%Y%m%d_%H%M%S).log"
else
    echo "ERROR: build.sh not found or not executable"
    exit 1
fi

# Check if build succeeded
IMAGE_TAG="${USER:-root}"
IMAGE_NAME="nvcr.io/nvstaging/tao/tao_pytorch_base_image:${IMAGE_TAG}"

if ! docker images | grep -q "nvstaging/tao/tao_pytorch_base_image"; then
    echo "ERROR: Docker build failed - image not found"
    exit 1
fi

echo ""
echo "=========================================="
echo "Docker build successful!"
echo "Image: $IMAGE_NAME"
echo "=========================================="

# Convert to SQSH if enroot is available
echo ""
echo "Step 3: Converting to SQSH format..."

if command -v enroot &> /dev/null; then
    echo "Enroot found, converting Docker image to SQSH..."
    
    # Create temporary directory for conversion
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"
    
    # Export and convert
    echo "Exporting Docker image..."
    docker save "$IMAGE_NAME" -o "${TEMP_DIR}/docker_image.tar"
    
    echo "Converting to SQSH..."
    enroot import --output "$OUTPUT_SQSH" "${TEMP_DIR}/docker_image.tar"
    
    # Cleanup
    rm -rf "$TEMP_DIR"
    
    echo ""
    echo "=========================================="
    echo "SQSH file created: $OUTPUT_SQSH"
    echo "File size: $(du -h "$OUTPUT_SQSH" | cut -f1)"
    echo "=========================================="
else
    echo "WARNING: enroot not found. Cannot convert to SQSH."
    echo "To use with SLURM/Pyxis, manually convert:"
    echo "  enroot import dockerd://$IMAGE_NAME"
    echo ""
    echo "Or use the Docker image directly on nodes with Docker."
fi

# List the built image
echo ""
echo "Built image details:"
docker images | grep -E "(nvstaging|REPOSITORY)" || true

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "Ended: $(date)"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Update your SLURM scripts to use:"
echo "   --container-image $OUTPUT_SQSH"
echo ""
echo "2. Or use Docker directly:"
echo "   docker run --gpus all -it --rm \\"
echo "     -v <CLUSTER_STORAGE>:<CLUSTER_STORAGE> \\"
echo "     -v /path/to/results:/workspace/results \\"
echo "     $IMAGE_NAME"
echo ""
