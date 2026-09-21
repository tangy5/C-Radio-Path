#!/bin/bash
# C-RADIO training starter. Reference spec: experiment_specs/
# c_radio_path_exp.yaml (teacher entries there are example
# configurations — substitute your own as needed).
#   Mode: combo (cosine summary + dampened_mse spatial), MLP 2048x2
#   LR 3e-5, warmup 2 ep, EMA 0.99998, fp16, 100 epochs, 8x GPU
#
# Usage:
#   ./start_training.sh                        # full run (8 GPUs)
#   NUM_GPUS=1 MAX_STEPS=10 ./start_training.sh  # 10-step smoke test
#
# Environment:
#   DATASET_ROOT  dir containing source_a_webdataset/ source_b_webdataset/
#                 source_c_webdataset/     (default: <kit>/data/webdatasets)
#   RESULTS_DIR   output dir                   (default: <kit>/run/results/c_radio_path_exp)
#   NUM_GPUS      GPUs to use                  (default: 8)
#   MAX_STEPS     stop after N steps, 0 = full (default: 0)

set -e
set -o pipefail   # training failures must propagate through the tee pipeline

NUM_GPUS="${NUM_GPUS:-8}"
MAX_STEPS="${MAX_STEPS:-0}"
PORT="${PORT:-29814}"   # kit default port
CONFIG_NAME="${CONFIG_NAME:-c_radio_path_exp}"   # spec in experiment_specs/ (without .yaml)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-$KIT_ROOT/data/webdatasets}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results/c_radio_path_exp}"
cd "$SCRIPT_DIR"

TEMPLATE="experiment_specs/${CONFIG_NAME}.yaml"
GENERATED="experiment_specs/generated.yaml"
[ -f "$TEMPLATE" ] || { echo "ERROR: missing $TEMPLATE"; exit 1; }

# ------------------------------------------------ path rewriting
sed -e "s|__KIT_ROOT__|$KIT_ROOT|g" \
    -e "s|__DATASET_ROOT__|$DATASET_ROOT|g" \
    -e "s|__RESULTS_DIR__|$RESULTS_DIR|g" \
    "$TEMPLATE" > "$GENERATED"
echo "Kit root:     $KIT_ROOT"
echo "Dataset root: $DATASET_ROOT"
echo "Results dir:  $RESULTS_DIR"

# ------------------------------------------------ resume handling
CHECKPOINT_DIR="$RESULTS_DIR/distill"
LATEST_CKPT=""
if [ -d "$CHECKPOINT_DIR" ]; then
    LATEST_CKPT=$(find "$CHECKPOINT_DIR" -name "*.pth" -not -name "*-EMA.pth" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
fi
if [ -n "$LATEST_CKPT" ]; then
    EPOCH=$(basename "$LATEST_CKPT" | grep -oP 'epoch_\K[0-9]+' || echo "0")
    echo "Resuming from $LATEST_CKPT (epoch $EPOCH)"
    sed -i "s|resume_training_checkpoint_path:.*|resume_training_checkpoint_path: $LATEST_CKPT|" "$GENERATED"
    [ "$EPOCH" -ge 100 ] && { echo "Training already complete."; exit 0; }
else
    echo "Fresh start."
fi

# ------------------------------------------------ verify inputs
GIGAPATH="$KIT_ROOT/teachers/GigaPath/pytorch_model.bin"
HOPTIMUS="$KIT_ROOT/teachers/H-optimus-0/pytorch_model.bin"
for f in "$HOPTIMUS"; do
    [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done
if [ ! -f "$GIGAPATH" ]; then
    echo "ERROR: Prov-GigaPath weights not found at: (referenced by the example teacher config)"
    echo "  $GIGAPATH"
    echo "Prov-GigaPath (example teacher) is NOT bundled. Obtain it via your own"
    echo "gated HuggingFace access and place it at the path above:"
    echo "  bash $KIT_ROOT/teachers/fetch_gigapath.sh"
    exit 1
fi

# verify every tar_data_sources root_dir from the GENERATED spec (supports
# custom customer webdatasets — any name, any location)
MISSING_SRC=0
while read -r srcdir; do
    [ -n "$srcdir" ] || continue
    n=$(find "$srcdir" -name '*.tar' 2>/dev/null | wc -l)
    echo "   $(basename "$srcdir"): $n shard files"
    [ "$n" -eq 0 ] && { echo "ERROR: no .tar shards under $srcdir"; MISSING_SRC=1; }
done < <(grep -A6 "tar_data_sources:" "$GENERATED" | grep "root_dir:" | sed 's/.*root_dir:[[:space:]]*//' | tr -d '"')
[ "$MISSING_SRC" -eq 0 ] || {
    echo "See data/MANIFEST.md for the shard lists this kit was trained with."
    exit 1
}

# ------------------------------------------------ fast-val dummy set
VAL_DIR="$KIT_ROOT/run/dataset/val_large/dummy_class"
if [ ! -d "$VAL_DIR" ] || [ "$(ls "$VAL_DIR" 2>/dev/null | wc -l)" -lt 64 ]; then
    echo "Generating 64 dummy fast-val images (limit_val_batches=2)..."
    python3 - "$VAL_DIR" <<'EOF'
import sys
from pathlib import Path
import numpy as np
from PIL import Image
d = Path(sys.argv[1]); d.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
for i in range(64):
    Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)).save(d / f"dummy_{i:03d}.png")
EOF
fi

# ------------------------------------------------ env / offline caches
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SCRIPT_DIR/.cache/torch" HF_HOME="$SCRIPT_DIR/.cache/hf"
mkdir -p "$TORCH_HOME" "$HF_HOME"
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export TAO_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=$(( $(nproc) / NUM_GPUS ))
export NVIDIA_TF32_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

TAO_REPO="$KIT_ROOT/code/tao-pytorch-C-Radio-Path-Exp"
export PYTHONPATH="$TAO_REPO:$TAO_REPO/tao-core:${PYTHONPATH:-}"

if [ -n "${PYTHON:-}" ] && [ -x "$PYTHON" ]; then :
else
    PYTHON="$(command -v python3)"
fi
echo "Python: $PYTHON | TAO repo: $TAO_REPO"

TRAINER_ARGS=""
[ "$MAX_STEPS" != "0" ] && TRAINER_ARGS="+trainer.max_steps=$MAX_STEPS"

mkdir -p "$RESULTS_DIR"
echo "Logs: tail -f $RESULTS_DIR/launcher.log"

$PYTHON -m torch.distributed.run \
    --nnodes=1 \
    --nproc-per-node=$NUM_GPUS \
    --master_port=$PORT \
    scripts/distill_fastval_ddp.py \
    --config-path="$SCRIPT_DIR/experiment_specs" \
    --config-name=generated \
    $TRAINER_ARGS \
    2>&1 | tee "$RESULTS_DIR/launcher.log"
