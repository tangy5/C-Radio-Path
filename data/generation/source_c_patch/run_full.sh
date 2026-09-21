#!/bin/bash
# Run full source C patch extraction pipeline with logging
# Usage: ./run_full.sh [dataset_name]  # optional: run specific dataset only

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
PROGRESS_FILE="${LOG_DIR}/progress.txt"

mkdir -p "$LOG_DIR"

echo "============================================" | tee "$LOG_FILE"
echo "source C Patch Extraction - Full Run" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

# Track progress
echo "0 / 60110 WSIs processed" > "$PROGRESS_FILE"

cd "$SCRIPT_DIR"

# Run pipeline, tee output to log
python run_pipeline.py "$@" 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "Completed: $(date)" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
