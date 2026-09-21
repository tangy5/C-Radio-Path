# freeze source A WebDataset Conversion Guide

## Overview

This guide explains how to convert the freeze source A dataset (1.6M image-text pairs) into WebDataset format with automatic resume capability for 4-hour runtime limits.

## Files Created

1. **`construct_webdataset_resumable.py`** - Main conversion script with resume capability
2. **`run_conversion_with_restart.sh`** - Wrapper script for automatic restart on 4-hour shutdown
3. **`conversion_state.pkl`** - State file (created automatically) for tracking progress

## Dataset Information

- **Input**: `<PATHOLOGY_DATASETS_DIR>/source_a_output/`
- **Output**: `<PATHOLOGY_DATASETS_DIR>/source_a_webdataset/`
- **Size**: ~1.6 TB input → ~1.7 TB output
- **Files**: ~1.6M image-text pairs
- **Estimated Time**: 6-15 hours (across multiple 4-hour sessions)

## Quick Start

### Option 1: Automatic Restart (Recommended)

Simply run the wrapper script and it will handle everything:

```bash
cd <CLUSTER_WORKSPACE>
./run_conversion_with_restart.sh
```

The script will:
- Run for 3.5 hours
- Gracefully save progress
- Automatically restart when the system comes back up
- Continue until all 1.6M samples are processed

### Option 2: Manual Control

If you prefer to manage the process manually:

```bash
# First run
python3 construct_webdataset_resumable.py \
    --input <PATHOLOGY_DATASETS_DIR>/source_a_output \
    --output <PATHOLOGY_DATASETS_DIR>/source_a_webdataset \
    --workers 16 \
    --shard-size 10000 \
    --checkpoint-interval 50000

# On restart (after shutdown), resume:
python3 construct_webdataset_resumable.py \
    --input <PATHOLOGY_DATASETS_DIR>/source_a_output \
    --output <PATHOLOGY_DATASETS_DIR>/source_a_webdataset \
    --resume \
    --workers 16
```

## How It Works

### Resume Mechanism

1. **State Tracking**: The script maintains a `conversion_state.pkl` file containing:
   - Which samples have been completed
   - Which shards have been written
   - Total samples count
   - Shuffled order of processing

2. **Checkpointing**: Every 50,000 samples (or your configured interval), the state is saved to disk.

3. **Graceful Shutdown**: When the 3.5-hour limit is reached (or SIGTERM received):
   - Current shard is closed properly
   - State is saved to disk
   - Process exits cleanly

4. **Automatic Resume**: On restart:
   - Script detects existing state file
   - Skips already-processed samples
   - Continues from where it left off

### Output Structure

```
source_a_webdataset/
├── 1000/
│   ├── shard_000000.tar   (10,000 samples)
│   ├── shard_000001.tar
│   ├── ...
│   └── shard_000160.tar   (161 total shards)
└── conversion_state.pkl   (deleted when complete)
```

Each `.tar` file contains triplets:
```
000000001.png      # image data
000000001.txt      # caption text
000000001.json     # {"sha256": "...", "caption": "..."}
```

## Configuration Options

### Main Script (`construct_webdataset_resumable.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | required | Input directory with image-text pairs |
| `--output` | required | Output directory for webdataset |
| `--shard-size` | 10000 | Samples per shard (tarfile) |
| `--max-per-subdir` | 1000 | Max tarfiles per subdirectory |
| `--workers` | num CPUs | Parallel worker processes |
| `--seed` | 42 | Random seed for shuffling |
| `--checkpoint-interval` | 50000 | Save state every N samples |
| `--state-file` | `conversion_state.pkl` | State file path |
| `--resume` | False | Resume from previous state |

### Wrapper Script (`run_conversion_with_restart.sh`)

| Option | Default | Description |
|--------|---------|-------------|
| `--input DIR` | source_a_output | Input directory |
| `--output DIR` | source_a_webdataset | Output directory |
| `--workers N` | 16 | Number of workers |
| `--shard-size N` | 10000 | Samples per shard |
| `--runtime-hours H` | 3.5 | Max hours before restart |

## Monitoring Progress

### During Runtime

The script shows a progress bar:
```
Processing samples:  45%|████▌    | 720000/1606000 [4:12:00<5:08:00, 47.8it/s]
```

### Check State File

```bash
python3 -c "
import pickle
with open('conversion_state.pkl', 'rb') as f:
    state = pickle.load(f)
    completed = len(state['completed_samples'])
    total = state['total_samples']
    print(f'Progress: {completed}/{total} ({100*completed/total:.1f}%)')
    print(f'Shards completed: {len(state[\"completed_shards\"])}')
    print(f'Last update: {state.get(\"last_update\", \"unknown\")}')
"
```

### Log Files

- `webdataset_conversion.log` - Main conversion logs
- `conversion_runner.log` - Wrapper script logs
- `conversion_state.pkl` - Binary state (for resume)

## Troubleshooting

### Process Was Killed Unexpectedly

The script saves state every 50,000 samples and at shutdown. If killed unexpectedly, you may lose at most 50,000 samples of progress. Just restart - it will resume from the last checkpoint.

### State File Corruption

If `conversion_state.pkl` is corrupted:
```bash
rm conversion_state.pkl
```
Then restart. The script will start fresh (you'll lose progress, but the output shards already written will remain - you'll need to manually delete them or use a new output directory).

### Disk Space Issues

If you run out of disk space:
1. Stop the conversion (Ctrl+C or wait for shutdown)
2. Free up space
3. Resume - the script will continue from where it left off

### Resume from Different Directory

If you move to a different system:
1. Copy the `conversion_state.pkl` file
2. Ensure the input data is at the same path (or modify the state file)
3. Run with `--resume`

## Completion

When all samples are processed:
- The script will print "All samples completed!"
- The `conversion_state.pkl` file is automatically deleted
- You'll have ~161 tar files in the `1000/` subdirectory

## Using the WebDataset

Once conversion is complete, use the dataset in PyTorch:

```python
import webdataset as wds

# Open the dataset
dataset = wds.WebDataset("<PATHOLOGY_DATASETS_DIR>/source_a_webdataset/1000/shard_{000000..000160}.tar")

# Decode samples
sample = wds.decode("pil")(dataset)

# Iterate
for sample in dataset:
    image = sample["png"]  # PIL Image
    caption = sample["txt"]  # str
    metadata = json.loads(sample["json"])
    print(f"SHA256: {metadata['sha256']}")
    print(f"Caption preview: {metadata['caption']}")
```

## Performance Tips

1. **Workers**: Use 16-32 workers for optimal throughput on Lustre
2. **Shard Size**: 10,000 is good balance between file count and shard size
3. **Checkpoint Interval**: Lower for more frequent saves (slower), higher for speed (more risk)
4. **Network**: Run during off-peak hours to avoid Lustre contention

## Estimated Timeline

| Samples | Time (16 workers) | Iterations (4h) |
|---------|-------------------|-----------------|
| 0-400K | ~3.5 hours | 1 |
| 400K-800K | ~3.5 hours | 1 |
| 800K-1200K | ~3.5 hours | 1 |
| 1200K-1606K | ~2.5 hours | 1 |
| **Total** | **~13 hours** | **4 iterations** |

Note: Actual times may vary based on system load and Lustre performance.
