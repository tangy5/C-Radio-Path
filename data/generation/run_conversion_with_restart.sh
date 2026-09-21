#!/bin/bash
#
# WebDataset Conversion Runner with Automatic Restart
# Handles 4-hour shutdown cycles by checkpointing and resuming
#
# Usage: ./run_conversion_with_restart.sh [options]
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERSION_SCRIPT="${SCRIPT_DIR}/construct_webdataset_resumable.py"
INPUT_DIR="<PATHOLOGY_DATASETS_DIR>/source_a_output"
OUTPUT_DIR="<PATHOLOGY_DATASETS_DIR>/source_a_webdataset"
STATE_FILE="${SCRIPT_DIR}/conversion_state.pkl"
LOG_FILE="${SCRIPT_DIR}/conversion_runner.log"
PID_FILE="${SCRIPT_DIR}/conversion.pid"

# Default parameters
SHARD_SIZE=10000
MAX_PER_SUBDIR=1000
WORKERS=16
SEED=42
CHECKPOINT_INTERVAL=50000
MAX_RUNTIME_HOURS=3.5  # Stop 30 min before 4-hour limit for safety

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --shard-size)
            SHARD_SIZE="$2"
            shift 2
            ;;
        --runtime-hours)
            MAX_RUNTIME_HOURS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --input DIR           Input directory with image-text pairs"
            echo "  --output DIR          Output directory for webdataset shards"
            echo "  --workers N           Number of worker processes (default: 16)"
            echo "  --shard-size N        Samples per shard (default: 10000)"
            echo "  --runtime-hours H     Max runtime before restart (default: 3.5)"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Cleanup function
cleanup() {
    log "Received shutdown signal. Cleaning up..."
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            log "Terminating conversion process (PID: $PID)..."
            kill -TERM "$PID" 2>/dev/null || true
            sleep 5
            kill -KILL "$PID" 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
    fi
    log "Cleanup complete. Ready for restart."
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Check if Python script exists
if [[ ! -f "$CONVERSION_SCRIPT" ]]; then
    log "ERROR: Conversion script not found: $CONVERSION_SCRIPT"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to run conversion with timeout
run_conversion() {
    local resume_flag=""
    
    # Check if we should resume
    if [[ -f "$STATE_FILE" ]]; then
        log "Found existing state file. Will resume conversion."
        resume_flag="--resume"
    else
        log "Starting fresh conversion."
    fi
    
    # Build command
    local cmd="python3 $CONVERSION_SCRIPT \
        --input $INPUT_DIR \
        --output $OUTPUT_DIR \
        --shard-size $SHARD_SIZE \
        --max-per-subdir $MAX_PER_SUBDIR \
        --workers $WORKERS \
        --seed $SEED \
        --checkpoint-interval $CHECKPOINT_INTERVAL \
        --state-file $STATE_FILE \
        $resume_flag"
    
    log "Starting conversion with command:"
    log "$cmd"
    log "Maximum runtime: ${MAX_RUNTIME_HOURS} hours"
    
    # Run with timeout (convert hours to seconds)
    local timeout_seconds=$(echo "$MAX_RUNTIME_HOURS * 3600" | bc | cut -d. -f1)
    
    # Start the process
    eval "$cmd" &
    local CONVERSION_PID=$!
    echo $CONVERSION_PID > "$PID_FILE"
    
    log "Conversion started with PID: $CONVERSION_PID"
    
    # Wait for process with timeout
    local waited=0
    while kill -0 $CONVERSION_PID 2>/dev/null; do
        sleep 10
        waited=$((waited + 10))
        
        # Check if we've exceeded max runtime
        if [[ $waited -ge $timeout_seconds ]]; then
            log "Maximum runtime (${MAX_RUNTIME_HOURS}h) reached. Initiating graceful shutdown..."
            kill -TERM $CONVERSION_PID 2>/dev/null || true
            
            # Wait for graceful shutdown
            local shutdown_wait=0
            while kill -0 $CONVERSION_PID 2>/dev/null && [[ $shutdown_wait -lt 60 ]]; do
                sleep 1
                shutdown_wait=$((shutdown_wait + 1))
            done
            
            # Force kill if still running
            if kill -0 $CONVERSION_PID 2>/dev/null; then
                log "Process didn't terminate gracefully. Force killing..."
                kill -KILL $CONVERSION_PID 2>/dev/null || true
            fi
            
            break
        fi
        
        # Progress update every 5 minutes
        if [[ $((waited % 300)) -eq 0 ]]; then
            local hours_elapsed=$(echo "scale=2; $waited / 3600" | bc)
            log "Progress: ${hours_elapsed}h elapsed, $(cat "$STATE_FILE" 2>/dev/null | python3 -c "import pickle,sys; d=pickle.load(sys.stdin.buffer); print(f'{len(d[\"completed_samples\"])}/{d[\"total_samples\"]}')" 2>/dev/null || echo 'state unavailable') samples completed"
        fi
    done
    
    # Clean up PID file
    rm -f "$PID_FILE"
    
    # Get exit status
    wait $CONVERSION_PID 2>/dev/null
    local exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        log "Conversion completed successfully!"
        return 0
    elif [[ $exit_code -eq 124 ]] || [[ $exit_code -eq 143 ]] || [[ $exit_code -eq 137 ]]; then
        log "Conversion was interrupted (timeout or signal). Will resume on restart."
        return 1
    else
        log "Conversion failed with exit code: $exit_code"
        return 2
    fi
}

# Check completion status
check_completion() {
    if [[ -f "$STATE_FILE" ]]; then
        local completed=$(python3 -c "
import pickle
with open('$STATE_FILE', 'rb') as f:
    state = pickle.load(f)
    print(len(state['completed_samples']))
" 2>/dev/null || echo "0")
        local total=$(python3 -c "
import pickle
with open('$STATE_FILE', 'rb') as f:
    state = pickle.load(f)
    print(state['total_samples'])
" 2>/dev/null || echo "1")
        
        if [[ "$completed" == "$total" ]] && [[ $completed -gt 0 ]]; then
            log "All samples completed ($completed/$total)!"
            return 0
        fi
    fi
    return 1
}

# Main execution loop
main() {
    log "=========================================="
    log "WebDataset Conversion Runner Started"
    log "Input: $INPUT_DIR"
    log "Output: $OUTPUT_DIR"
    log "Workers: $WORKERS"
    log "Max Runtime: ${MAX_RUNTIME_HOURS}h"
    log "=========================================="
    
    # Check if already complete
    if check_completion; then
        log "Conversion already complete. Nothing to do."
        exit 0
    fi
    
    # Run conversion loop
    local iteration=0
    while true; do
        iteration=$((iteration + 1))
        log "=========================================="
        log "Starting iteration $iteration"
        log "=========================================="
        
        run_conversion
        local status=$?
        
        if [[ $status -eq 0 ]]; then
            log "SUCCESS: Conversion complete!"
            break
        elif [[ $status -eq 1 ]]; then
            log "Conversion interrupted. Preparing for restart..."
            log "Waiting 10 seconds before checking state..."
            sleep 10
            
            # Check if we're done
            if check_completion; then
                log "All samples completed during previous run!"
                break
            fi
            
            log "State shows incomplete conversion. Will restart automatically..."
            log "Waiting 30 seconds for system to stabilize..."
            sleep 30
            
            # Check state file exists
            if [[ ! -f "$STATE_FILE" ]]; then
                log "ERROR: State file missing! Cannot resume."
                exit 1
            fi
            
            # Show progress
            local progress=$(python3 -c "
import pickle
with open('$STATE_FILE', 'rb') as f:
    state = pickle.load(f)
    completed = len(state['completed_samples'])
    total = state['total_samples']
    pct = 100.0 * completed / total if total > 0 else 0
    print(f'{completed}/{total} ({pct:.1f}%)')
" 2>/dev/null || echo "unknown")
            log "Current progress: $progress"
            log "Restarting conversion..."
        else
            log "ERROR: Conversion failed. Check logs for details."
            exit 1
        fi
    done
    
    log "=========================================="
    log "Conversion process finished successfully!"
    log "Output location: $OUTPUT_DIR"
    log "Total iterations: $iteration"
    log "=========================================="
}

# Run main
main "$@"
