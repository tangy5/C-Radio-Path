#!/bin/bash
# Convert all remaining source C PNG patches to JPEG - ALL DATASETS IN PARALLEL.
# Each dataset gets 56 workers (total ~392 across 7 datasets, well within 128 vCPU / 2TB RAM).
# This maximizes throughput by saturating both CPU and I/O bandwidth.

SCRIPT="<PATHOLOGY_DATASETS_DIR>/source_c_patch/convert_png_to_jpeg.py"
LOGDIR="<PATHOLOGY_DATASETS_DIR>/source_c_patch/logs"

mkdir -p "$LOGDIR"

DATASETS=(
    "source C-gastrointestinal"
    "source C-hematologic"
    "source C-thorax"
    "source C-breast"
    "source C-colorectal-b1"
    "source C-skin-b1"
    "source C-skin-b2"
)

WORKERS=56

echo "============================================"
echo "source C PNG to JPEG Parallel Conversion"
echo "Started: $(date)"
echo "Datasets: ${#DATASETS[@]} in parallel"
echo "Workers: $WORKERS per dataset"
echo "============================================"

PIDS=()

for ds in "${DATASETS[@]}"; do
    logfile="$LOGDIR/convert_${ds#source C-}_$(date +%Y%m%d_%H%M%S).log"
    echo ">>> Starting: $ds (PID logging to $logfile) at $(date)"
    python "$SCRIPT" --dataset "$ds" --workers "$WORKERS" --quality 95 > "$logfile" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All $(( ${#PIDS[@]} )) conversion processes launched."
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
echo "All conversions complete at $(date)"
echo "Failed: $FAILED / ${#DATASETS[@]}"
echo "============================================"
