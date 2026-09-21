#!/bin/bash
# Quick build script for TAO PyTorch Docker
# Run this on a node with Docker access

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}TAO PyTorch Docker Quick Build${NC}"
echo -e "${GREEN}==========================================${NC}"

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR: Docker not found${NC}"
    echo "Please install Docker or run on a node with Docker access"
    exit 1
fi

echo -e "${GREEN}✓ Docker found${NC}"

# Check docker daemon
if ! docker info &> /dev/null; then
    echo -e "${RED}ERROR: Docker daemon not accessible${NC}"
    echo "You may need to:"
    echo "  1. Start Docker: sudo systemctl start docker"
    echo "  2. Add yourself to docker group: sudo usermod -a -G docker \$(whoami)"
    echo "  3. Log out and log back in"
    exit 1
fi

echo -e "${GREEN}✓ Docker daemon accessible${NC}"

# Get repository path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source environment
echo ""
echo -e "${YELLOW}Setting up environment...${NC}"
source scripts/envsetup.sh

if [ -z "${NV_TAO_PYTORCH_TOP:-}" ]; then
    echo -e "${RED}ERROR: Failed to set NV_TAO_PYTORCH_TOP${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Environment set: $NV_TAO_PYTORCH_TOP${NC}"

# Build options
BUILD_PLATFORM="${BUILD_PLATFORM:-linux/amd64}"
FORCE_BUILD="${FORCE_BUILD:-0}"

echo ""
echo -e "${YELLOW}Build configuration:${NC}"
echo "  Platform: $BUILD_PLATFORM"
echo "  Force rebuild: $FORCE_BUILD"
echo ""

# Navigate to docker directory
cd "$NV_TAO_PYTORCH_TOP/docker"

# Check if xformers directory exists (from previous failed build)
if [ -d "xformers" ]; then
    echo -e "${YELLOW}Removing old xformers directory...${NC}"
    rm -rf xformers
fi

# Build
echo -e "${YELLOW}Starting Docker build...${NC}"
echo -e "${YELLOW}This will take 1-3 hours. Grab a coffee!${NC}"
echo ""

BUILD_ARGS="--build"

if [ "$BUILD_PLATFORM" = "linux/amd64" ] || [ "$BUILD_PLATFORM" = "x86" ]; then
    BUILD_ARGS="$BUILD_ARGS --x86"
elif [ "$BUILD_PLATFORM" = "linux/arm64" ] || [ "$BUILD_PLATFORM" = "arm" ]; then
    BUILD_ARGS="$BUILD_ARGS --arm"
fi

if [ "$FORCE_BUILD" = "1" ]; then
    BUILD_ARGS="$BUILD_ARGS --force"
fi

echo "Running: ./build.sh $BUILD_ARGS"
echo ""

# Run build with logging
LOG_FILE="docker_build_$(date +%Y%m%d_%H%M%S).log"
./build.sh $BUILD_ARGS 2>&1 | tee "$LOG_FILE"

# Check result
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}==========================================${NC}"
    echo -e "${GREEN}Build successful!${NC}"
    echo -e "${GREEN}==========================================${NC}"
    
    IMAGE_TAG="${USER:-$(whoami)}"
    IMAGE_NAME="nvcr.io/nvstaging/tao/tao_pytorch_base_image:${IMAGE_TAG}"
    
    echo ""
    echo "Built image: $IMAGE_NAME"
    echo ""
    echo "To use this image:"
    echo ""
    echo "1. Source the environment:"
    echo "   source $SCRIPT_DIR/scripts/envsetup.sh"
    echo ""
    echo "2. Run interactive container:"
    echo "   tao_pt --gpus all --volume <CLUSTER_STORAGE>:<CLUSTER_STORAGE>"
    echo ""
    echo "3. Or with Docker directly:"
    echo "   docker run --gpus all -it --rm \\"
    echo "     -v <CLUSTER_STORAGE>:<CLUSTER_STORAGE> \\"
    echo "     -v $SCRIPT_DIR:/tao-pt \\"
    echo "     -e PYTHONPATH=/tao-pt \\"
    echo "     --shm-size=16g \\"
    echo "     $IMAGE_NAME"
    echo ""
    
else
    echo ""
    echo -e "${RED}==========================================${NC}"
    echo -e "${RED}Build failed!${NC}"
    echo -e "${RED}==========================================${NC}"
    echo ""
    echo "Check the log file: $LOG_FILE"
    echo ""
    echo "Common issues:"
    echo "  - Disk space: Run 'docker system prune -a'"
    echo "  - Network: Ensure you can reach nvcr.io"
    echo "  - Permissions: Ensure you're in docker group"
    exit 1
fi
