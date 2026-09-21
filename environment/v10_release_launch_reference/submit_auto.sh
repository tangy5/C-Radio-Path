#!/bin/bash
# Auto-submit chained Slurm jobs for V10 Hyena training
# Chains multiple 4-hour jobs until training completes (100 epochs)
# Mirrors v9.2's submit_auto.sh exactly, adapted for v10
#
# Student: hyena_vit_g_last4 (ViT-G/14 with Hyena in blocks 36-39)
# Teachers: per experiment spec (example configuration)
#
# Usage:
#   ./submit_auto.sh [EXPERIMENT_NAME] [MAX_JOBS] [PARTITION]
#   ./submit_auto.sh                                          # defaults
#   ./submit_auto.sh v10_hyena_last4_vitg_2teacher 25 batch

set -e

# Configuration
WORKSPACE_DIR="<CLUSTER_WORKSPACE>"
EXPERIMENT_NAME="${1:-v10_hyena_last4_vitg_2teacher}"
MAX_JOBS="${2:-3}"
PARTITION="${3:-batch}"
MAX_EPOCHS=100

CONTAINER_IMAGE="$WORKSPACE_DIR/tao_dlmed_pathology_base_yt_v1.sqsh"
EXP_DIR="$WORKSPACE_DIR/v10"

echo "==============================================="
echo "Auto-Submit C-RADIO V10 Hyena Training Jobs"
echo "   Student: hyena_vit_g_last4 (ViT-G/14, Hyena blocks 36-39)"
echo "   Teachers: per experiment spec"
echo "   Mode: combo (cosine summary + dampened_mse spatial)"
echo "   Dataset: source A + source B + source C (~13.17M images)"
echo "   LR: 3e-5, warmup: 2 epochs, epochs: 100"
echo "==============================================="
echo ""
echo "Experiment: $EXPERIMENT_NAME"
echo "Partition: $PARTITION"
echo "Max jobs: $MAX_JOBS"
echo "Container: $CONTAINER_IMAGE"
echo ""

# Verify container exists
if [ ! -f "$CONTAINER_IMAGE" ]; then
    echo "ERROR: Container not found: $CONTAINER_IMAGE"
    exit 1
fi

# Change to workspace
cd "$WORKSPACE_DIR"
mkdir -p "$EXP_DIR/slurm_logs"

# Function to get current epoch from checkpoint
get_current_epoch() {
    local results_dir="$EXP_DIR/results/v10_hyena_last4/distill"

    if [ -d "$results_dir" ]; then
        local latest=""
        for ckpt in $(find "$results_dir" -name "*.pth" -not -name "*-EMA.pth" -not -name "classifier_model_*" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | cut -d' ' -f2-); do
            local ema_ckpt="${ckpt%.pth}-EMA.pth"
            if [ -f "$ema_ckpt" ]; then
                latest="$ckpt"
                break
            fi
        done
        if [ -n "$latest" ]; then
            basename "$latest" | grep -oP 'epoch_\K[0-9]+' || echo "0"
            return
        fi
    fi
    echo "0"
}

# Check initial status
CURRENT_EPOCH=$(( 10#$(get_current_epoch) ))
echo "Current epoch: $CURRENT_EPOCH"

if [ "$CURRENT_EPOCH" -ge $MAX_EPOCHS ]; then
    echo "Training already complete!"
    exit 0
fi

echo ""
echo "Submitting job chain..."
echo ""

# Function to find last job in existing dependency chain
find_last_job_in_chain() {
    local exp_name="$1"

    local pending_ids=$(squeue -u "$USER" -n "$exp_name" -t PENDING -h -o "%i" 2>/dev/null | tr -d ' ' | sort -n)
    if [ -n "$pending_ids" ]; then
        echo "$pending_ids" | tail -1
        return
    fi

    local running_ids=$(squeue -u "$USER" -n "$exp_name" -t RUNNING -h -o "%i" 2>/dev/null | tr -d ' ' | sort -n)
    if [ -n "$running_ids" ]; then
        echo "$running_ids" | tail -1
        return
    fi

    echo ""
}

# Function to submit a job
submit_job() {
    local dep_id="$1"
    local job_name="v10_hyena_${EXPERIMENT_NAME}"

    local sbatch_args=""
    if [ -n "$dep_id" ]; then
        sbatch_args="--dependency=afterany:$dep_id"
    fi

    local new_job_id=$(sbatch \
        $sbatch_args \
        --job-name="$job_name" \
        --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=128 \
        --gres=gpu:8 \
        --partition="$PARTITION" \
        --account=healthcareeng_monai \
        --time=4:00:00 \
        --output="$EXP_DIR/slurm_logs/%x_%j.out" \
        --error="$EXP_DIR/slurm_logs/%x_%j.err" \
        --export=ALL,CONFIG_NAME="$EXPERIMENT_NAME",NUM_GPUS=8,MAX_STEPS=0 \
        --wrap="
            srun --container-image=$CONTAINER_IMAGE --container-mounts=/home/$USER:/home/$USER,<CLUSTER_STORAGE>:<CLUSTER_STORAGE> bash -c '
                cd $EXP_DIR && \
                export CONFIG_NAME=$EXPERIMENT_NAME && \
                export NUM_GPUS=8 && \
                export MAX_STEPS=0 && \
                ./start_training.sh
            '
        " \
        | awk '{print $4}')

    echo "$new_job_id"
}

# Check for existing job chain and submit
FULL_JOB_NAME="v10_hyena_${EXPERIMENT_NAME}"
LAST_JOB_ID=$(find_last_job_in_chain "$FULL_JOB_NAME")

if [ -n "$LAST_JOB_ID" ]; then
    echo "Found existing job chain, last job: $LAST_JOB_ID"
    echo "Chaining new jobs after it..."
    FIRST_DEP="$LAST_JOB_ID"
else
    echo "No existing jobs found, starting fresh chain..."
    FIRST_DEP=""
fi

# Submit first job (with or without dependency)
JOB_ID=$(submit_job "$FIRST_DEP")
if [ -n "$FIRST_DEP" ]; then
    echo "   Job 1: $JOB_ID (depends on $FIRST_DEP)"
else
    echo "   Job 1: $JOB_ID (Epoch $CURRENT_EPOCH)"
fi

JOBS_SUBMITTED=1

# Chain remaining jobs
while [ "$JOBS_SUBMITTED" -lt "$MAX_JOBS" ]; do
    sleep 1

    NEXT_JOB_ID=$(submit_job "$JOB_ID")

    JOBS_SUBMITTED=$((JOBS_SUBMITTED + 1))
    echo "   Job $JOBS_SUBMITTED: $NEXT_JOB_ID (depends on $JOB_ID)"

    JOB_ID=$NEXT_JOB_ID

    # Rough estimate: ~9h per epoch (same as v9.2, 2 fewer teachers but Hyena overhead)
    EST_EPOCHS=$((CURRENT_EPOCH + (JOBS_SUBMITTED * 4 / 9)))
    if [ "$EST_EPOCHS" -ge $MAX_EPOCHS ]; then
        echo ""
        echo "Estimated completion at ~$EST_EPOCHS epochs"
        break
    fi
done

echo ""
echo "==============================================="
echo "Job chain submitted!"
echo "==============================================="
echo "Total jobs: $JOBS_SUBMITTED"
echo ""
echo "Monitor with:"
echo "   squeue -u \$USER"
echo ""
echo "Cancel all jobs:"
echo "   scancel -u \$USER -n v10_hyena_${EXPERIMENT_NAME}"
