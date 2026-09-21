#!/bin/bash
# source C Patch Quality Check - All datasets in parallel.
# 8 datasets x 16 workers = 128 processes (matches 128 vCPU).
# Resumable: re-run will skip already-checked directories.

SCRIPT="<PATHOLOGY_DATASETS_DIR>/source_c_patch/check_patch_quality.py"
LOGDIR="<PATHOLOGY_DATASETS_DIR>/source_c_patch/logs"

mkdir -p "$LOGDIR"

DATASETS=(
    "source C-breast"
    "source C-colorectal-b1"
    "source C-colorectal-b2"
    "source C-gastrointestinal"
    "source C-hematologic"
    "source C-skin-b1"
    "source C-skin-b2"
    "source C-thorax"
)

WORKERS=16

echo "============================================"
echo "source C Patch Quality Check - Parallel Run"
echo "Started: $(date)"
echo "Datasets: ${#DATASETS[@]} in parallel"
echo "Workers: $WORKERS per dataset"
echo "Total processes: $((${#DATASETS[@]} * WORKERS))"
echo "============================================"

PIDS=()

for ds in "${DATASETS[@]}"; do
    logfile="$LOGDIR/qc_${ds#source C-}_$(date +%Y%m%d_%H%M%S).log"
    echo ">>> Starting: $ds (logging to $logfile) at $(date)"
    python "$SCRIPT" --dataset "$ds" --workers "$WORKERS" > "$logfile" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All ${#PIDS[@]} quality check processes launched."
echo "PIDs: ${PIDS[*]}"
echo ""

# Wait for all and report
FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    EXIT=$?
    ds="${DATASETS[$i]}"
    if [ $EXIT -eq 0 ]; then
        echo ">>> DONE: $ds at $(date)"
    else
        echo ">>> FAILED (exit $EXIT): $ds at $(date)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================"
echo "All quality checks complete at $(date)"
echo "Failed: $FAILED / ${#DATASETS[@]}"
echo "============================================"

# Print summary report
echo ""
echo "Generating summary report..."
python "$SCRIPT" --report-only --output-dir "$LOGDIR"
