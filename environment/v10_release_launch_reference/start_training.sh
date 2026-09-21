#!/bin/bash
# V10 Experiment: Hyena ViT-G/14 (last 4 blocks) - Dual Teacher Distillation
# Training launcher — handles both fresh start and resume
#
# Based on v9.2's start_training.sh — same env, same container, same patterns
#
# Student: hyena_vit_g_last4 (ViT-G/14 with Hyena replacing attention in blocks 36-39)
# Teachers: per experiment spec (example dual-teacher configuration)
# Dataset: source A + source B + source C (~13.17M images)
# LR: 3e-5, warmup: 2 epochs, EMA: 0.99998, epochs: 100
#
# For quick test: set NUM_GPUS=1 and MAX_STEPS=10
# For full run:   set NUM_GPUS=8 and MAX_STEPS=0 (unlimited)

set -e

echo "==============================================="
echo "C-RADIO V10 Hyena ViT-G Training (Dual Teacher)"
echo "  Student: hyena_vit_g_last4 (ViT-G/14, Hyena blocks 36-39)"
echo "  Teachers: per experiment spec"
echo "  Mode: combo (cosine summary + dampened_mse spatial)"
echo "  Full fine-tune: backbone unfrozen (~1.1B trainable)"
echo "  LR: 3e-5, warmup: 2 epochs, EMA: 0.99998"
echo "  Dataset: source A + source B + source C (~13.17M images)"
echo "==============================================="

# Configuration — can be overridden via environment variables
CONFIG_NAME="${CONFIG_NAME:-v10_hyena_last4_vitg_2teacher}"
NUM_GPUS="${NUM_GPUS:-1}"
MAX_STEPS="${MAX_STEPS:-10}"   # 0 = unlimited (full training)

# Ensure we're in the correct directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SPEC_FILE="experiment_specs/${CONFIG_NAME}.yaml"

# Verify spec file exists
if [ ! -f "$SPEC_FILE" ]; then
    echo "ERROR: Spec file not found: $SPEC_FILE"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Extract results_dir from YAML file
RESULTS_DIR=$(grep "^results_dir:" "$SPEC_FILE" | head -1 | awk '{print $2}' | tr -d '[:space:]')

echo "Configuration: $CONFIG_NAME"
echo "Results directory: $RESULTS_DIR"
echo "Num GPUs: $NUM_GPUS"
if [ "$MAX_STEPS" != "0" ]; then
    echo "Quick test mode: max_steps=$MAX_STEPS"
fi

# ==============================================
# Environment setup — matches v9.2 exactly
# ==============================================

# TAO codebase (shared TAO codebase)
export TAO_REPO=<CLUSTER_WORKSPACE>/v10/tao-pytorch-C-Radio-Path-Exp
export PYTHONPATH="$TAO_REPO:$TAO_REPO/tao-core:$SCRIPT_DIR:$SCRIPT_DIR/nvsubquadratic:$PYTHONPATH"

# Python: kit conda env (see environment/README.md)
CONDA_ENV=<CONDA_ENVS_DIR>/tao_v7_py310
if [ -f "$CONDA_ENV/bin/python" ]; then
    export PYTHON="$CONDA_ENV/bin/python"
    echo "Using conda env Python: $PYTHON"
else
    export PYTHON="$(which python3)"
    echo "WARNING: Using host Python: $PYTHON"
fi

echo "Using TAO codebase: $TAO_REPO"
echo "Student backbone: hyena_vit_g_last4 (ViT-G/14, 1536 dim, Hyena in blocks 36-39)"

# HuggingFace offline mode (same as v9.2)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SCRIPT_DIR/.cache/torch"
export HF_HOME="$SCRIPT_DIR/.cache/hf"
export TRANSFORMERS_CACHE="$SCRIPT_DIR/.cache/transformers"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$TRANSFORMERS_CACHE"

# GPU config (same as v9.2)
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export TAO_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=$(( $(nproc) / NUM_GPUS ))
export NVIDIA_TF32_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "TAO_VISIBLE_DEVICES: $TAO_VISIBLE_DEVICES"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"

# Create results directory
mkdir -p "$RESULTS_DIR"

# ==============================================
# STEP 1: Check for existing checkpoint (resume logic — same as v9.2)
# ==============================================
echo ""
echo "Checking for existing checkpoints..."

CHECKPOINT_DIR="$RESULTS_DIR/distill"
LATEST_CKPT=""

if [ -d "$CHECKPOINT_DIR" ]; then
    # Find latest checkpoint that has BOTH .pth and -EMA.pth (EMA required for resume)
    for ckpt in $(find "$CHECKPOINT_DIR" -name "*.pth" -not -name "*-EMA.pth" -not -name "classifier_model_*" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | cut -d' ' -f2-); do
        ema_ckpt="${ckpt%.pth}-EMA.pth"
        if [ -f "$ema_ckpt" ]; then
            LATEST_CKPT="$ckpt"
            break
        fi
    done
fi

# ==============================================
# STEP 2: Determine start mode
# ==============================================
MAX_EPOCHS=100
if [ -n "$LATEST_CKPT" ]; then
    EPOCH=$(basename "$LATEST_CKPT" | grep -oP 'epoch_\K[0-9]+' || echo "0")
    echo "Found checkpoint: $(basename "$LATEST_CKPT")"
    echo "Resuming from Epoch: $EPOCH"
    echo ""

    if [ "$EPOCH" -ge $MAX_EPOCHS ]; then
        echo "Training already complete (Epoch $EPOCH)!"
        exit 0
    fi

    sed -i "s|resume_training_checkpoint_path:.*|resume_training_checkpoint_path: $LATEST_CKPT|" "$SPEC_FILE"
    START_MODE="resume"
else
    echo "No checkpoint found. Starting fresh training..."
    echo ""

    sed -i "s|resume_training_checkpoint_path:.*|resume_training_checkpoint_path: null|" "$SPEC_FILE"
    START_MODE="fresh"
fi

# ==============================================
# STEP 3: Verify teacher & student models (same as v9.2)
# ==============================================
echo "Verifying models..."
GIGAPATH_MODEL=<CLUSTER_WORKSPACE>/teacher_models/GigaPath/pytorch_model.bin
HOPTIMUS_MODEL=<CLUSTER_WORKSPACE>/teacher_models/H-optimus-0/pytorch_model.bin
INIT_BACKBONE="$SCRIPT_DIR/v10_hyena_backbone.pth"

if [ -f "$GIGAPATH_MODEL" ]; then
    echo "   Prov-GigaPath weights: OK ($(du -sh $GIGAPATH_MODEL | cut -f1))"
else
    echo "   ERROR: Prov-GigaPath model not found: $GIGAPATH_MODEL"
    exit 1
fi
if [ -f "$HOPTIMUS_MODEL" ]; then
    echo "   H-optimus-0 weights: OK ($(du -sh $HOPTIMUS_MODEL | cut -f1))"
else
    echo "   ERROR: H-optimus-0 model not found: $HOPTIMUS_MODEL"
    exit 1
fi
if [ -f "$INIT_BACKBONE" ]; then
    echo "   v10 Hyena backbone (student pretrained): OK ($(du -sh $INIT_BACKBONE | cut -f1))"
else
    echo "   ERROR: v10 Hyena backbone not found: $INIT_BACKBONE"
    exit 1
fi
echo ""

# ==============================================
# STEP 4: Verify datasets (same as v9.2)
# ==============================================
echo "Verifying datasets..."
SOURCE_A_SHARDS=$(ls <PATHOLOGY_DATASETS_DIR>/source_a_webdataset/1000/*.tar 2>/dev/null | wc -l)
echo "   source A: $SOURCE_A_SHARDS shard files"

SOURCE_B_SHARDS=$(ls <PATHOLOGY_DATASETS_DIR>/source_b_webdataset/1000/*.tar 2>/dev/null | wc -l)
echo "   source B: $SOURCE_B_SHARDS shard files"

SOURCE_C_SHARDS=$(find <PATHOLOGY_DATASETS_DIR>/source_c_webdataset -name "*.tar" 2>/dev/null | wc -l)
echo "   source C: $SOURCE_C_SHARDS shard files"
echo "   Total: $((SOURCE_A_SHARDS + SOURCE_B_SHARDS + SOURCE_C_SHARDS)) shards (~13.17M images)"
echo ""

# ==============================================
# STEP 5: Launch Training (same launcher as v9.2)
# ==============================================
echo "==============================================="
if [ "$START_MODE" = "resume" ]; then
    echo "LAUNCHING: Resume Training (Epoch $EPOCH)"
else
    echo "LAUNCHING: Fresh Training (Epoch 0)"
fi
echo "==============================================="
echo ""
echo "Configuration:"
echo "   Student: hyena_vit_g_last4 (ViT-G/14, 1536 dim, 40 layers, Hyena blocks 36-39)"
echo "   Student pretrained: per experiment spec"
echo "   Full fine-tune: backbone unfrozen (~1.1B trainable)"
echo "   Teacher entries: per experiment spec (example configuration)"

echo "   Mode: combo (cosine summary + dampened_mse spatial)"
echo "   MLP: 2048 hidden, 2 inner blocks"
echo "   GPUs: ${NUM_GPUS}x $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "   Batch size: 32 per GPU ($((32 * NUM_GPUS)) global)"
echo "   LR: 3e-5, warmup: 2 epochs, EMA decay: 0.99998"
echo "   Epochs: 100"
echo "   Checkpoint: milestone_keep=7, latest every 2000 steps"
echo "   DDP master port: 29816"
if [ "$MAX_STEPS" != "0" ]; then
    echo "   QUICK TEST: max_steps=$MAX_STEPS"
fi
echo ""
echo "Logs: tail -f $RESULTS_DIR/launcher.log"
echo "==============================================="
echo ""

# Build trainer_args for max_steps limit (quick test)
TRAINER_ARGS=""
if [ "$MAX_STEPS" != "0" ]; then
    TRAINER_ARGS="+trainer.max_steps=$MAX_STEPS"
fi

# Run training — using distill_fastval_ddp.py (same as v9.2, no schema validation needed)
# DDP master port: 29816 (distinct from v9.2's 29814 and v9.3's 29815 to allow simultaneous runs)
$PYTHON -m torch.distributed.run \
    --nnodes=1 \
    --nproc-per-node=$NUM_GPUS \
    --master_port=29816 \
    scripts/distill_fastval_ddp.py \
    --config-path="$SCRIPT_DIR/experiment_specs" \
    --config-name="$CONFIG_NAME" \
    $TRAINER_ARGS \
    2>&1 | tee "$RESULTS_DIR/launcher.log"

echo ""
echo "==============================================="
echo "Training session completed"
echo "==============================================="
