# Dataloader Visualization with Hydra

The visualization script now uses `hydra_runner` to properly load configs with schema defaults. This ensures `mean` and `std` are always available.

## Quick Start

### Basic Usage

```bash
cd <EDGEAI_ROOT>/users/<user>/workspace/tmp/tao-pytorch-all/nvidia_tao_pytorch/cv/classification_pyt

# Visualize using debug2.yaml config
python scripts/viz_simple.py --config-name debug2

# With custom output directory
VIZ_OUTPUT_DIR=./my_viz python scripts/viz_simple.py --config-name debug2

# With custom parameters
VIZ_OUTPUT_DIR=./viz \
VIZ_NUM_BATCHES=5 \
VIZ_IMAGES_PER_BATCH=16 \
python scripts/viz_simple.py --config-name debug2
```

## Environment Variables

Control visualization parameters via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `VIZ_OUTPUT_DIR` | `./dataloader_viz` | Output directory for images |
| `VIZ_NUM_BATCHES` | `3` | Number of batches to save |
| `VIZ_IMAGES_PER_BATCH` | `8` | Images per batch |

## Hydra Features

### Override Config Values

Use Hydra's override syntax:

```bash
# Override batch size
python scripts/viz_simple.py --config-name debug2 \
    dataset.batch_size=4

# Override augmentation settings
python scripts/viz_simple.py --config-name debug2 \
    dataset.augmentation.random_flip.hflip_probability=1.0

# Override multiple values
python scripts/viz_simple.py --config-name debug2 \
    dataset.batch_size=8 \
    dataset.img_size=384 \
    dataset.augmentation.with_random_crop=true
```

### Use Different Config Files

```bash
# Use a different config file in experiment_specs/
python scripts/viz_simple.py --config-name my_experiment

# Use config from different directory
python scripts/viz_simple.py --config-path /path/to/configs --config-name my_config
```

## Complete Examples

### Example 1: Basic Visualization

```bash
cd <EDGEAI_ROOT>/users/<user>/workspace/tmp/tao-pytorch-all/nvidia_tao_pytorch/cv/classification_pyt

python scripts/viz_simple.py --config-name debug2
```

Output:
```
Output directory: ./dataloader_viz
Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
Visualizing 3 batches, 8 images per batch

Saving images to ./dataloader_viz...
  Batch 0: saved 8 images
  Batch 1: saved 8 images
  Batch 2: saved 8 images
✅ Done! Check ./dataloader_viz
```

### Example 2: Custom Output and Parameters

```bash
VIZ_OUTPUT_DIR=/tmp/viz_output \
VIZ_NUM_BATCHES=5 \
VIZ_IMAGES_PER_BATCH=16 \
python scripts/viz_simple.py --config-name debug2
```

### Example 3: Test Different Augmentations

```bash
# Visualize without random flip
VIZ_OUTPUT_DIR=./viz_no_flip \
python scripts/viz_simple.py --config-name debug2 \
    dataset.augmentation.random_flip.enable=false

# Visualize with random crop
VIZ_OUTPUT_DIR=./viz_with_crop \
python scripts/viz_simple.py --config-name debug2 \
    dataset.augmentation.with_random_crop=true

# Visualize with different image size
VIZ_OUTPUT_DIR=./viz_384 \
python scripts/viz_simple.py --config-name debug2 \
    dataset.img_size=384
```

### Example 4: Test Different Resolutions

```bash
# Test multi-scale augmentation
VIZ_OUTPUT_DIR=./viz_multiscale \
VIZ_NUM_BATCHES=10 \
python scripts/viz_simple.py --config-name debug2 \
    dataset.augmentation.multi_scales='[{224: 0.3}, {256: 0.4}, {288: 0.3}]'
```

## Why Use Hydra Runner?

### Before (Direct OmegaConf)

```python
# Problem: Missing defaults
config = OmegaConf.load("debug2.yaml")
mean = config.dataset.augmentation.get("mean")  # None - not in YAML!
```

### After (Hydra Runner with Schema)

```python
# Solution: Schema defaults are merged automatically
@hydra_runner(schema=ExperimentConfig)
def main(cfg: ExperimentConfig):
    mean = cfg.dataset.augmentation.mean  # [0.485, 0.456, 0.406] - from schema!
```

### Benefits

✅ **Automatic defaults** - All schema defaults are available  
✅ **Type validation** - Config is validated against schema  
✅ **Override support** - Easy command-line overrides  
✅ **Consistency** - Same loading as training script  
✅ **No missing values** - `mean` and `std` always present  

## Advanced Usage

### Programmatic Usage

If you want to call the visualization from Python code:

```python
import os
os.environ['VIZ_OUTPUT_DIR'] = './my_output'
os.environ['VIZ_NUM_BATCHES'] = '5'

# Call the script
os.system('python scripts/viz_simple.py --config-name debug2')
```

### Batch Script

Create a shell script to visualize multiple configs:

```bash
#!/bin/bash
# visualize_all.sh

for config in debug2 experiment1 experiment2; do
    echo "Visualizing $config..."
    VIZ_OUTPUT_DIR="./viz_${config}" \
    VIZ_NUM_BATCHES=3 \
    python scripts/viz_simple.py --config-name "$config"
done
```

### Compare Augmentations Side-by-Side

```bash
# Strong augmentation
VIZ_OUTPUT_DIR=./aug_strong \
python scripts/viz_simple.py --config-name debug2 \
    dataset.augmentation.random_color.brightness=0.8 \
    dataset.augmentation.random_color.contrast=0.8

# Weak augmentation  
VIZ_OUTPUT_DIR=./aug_weak \
python scripts/viz_simple.py --config-name debug2 \
    dataset.augmentation.random_color.brightness=0.2 \
    dataset.augmentation.random_color.contrast=0.2

# No augmentation
VIZ_OUTPUT_DIR=./aug_none \
python scripts/viz_simple.py --config-name debug2 \
    dataset.augmentation.random_flip.enable=false \
    dataset.augmentation.random_color.enable=false
```

## Troubleshooting

### Issue: Config not found

```
Error: Could not find config file 'debug2.yaml'
```

**Solution:** Make sure you're in the correct directory and config exists:
```bash
cd <EDGEAI_ROOT>/users/<user>/workspace/tmp/tao-pytorch-all/nvidia_tao_pytorch/cv/classification_pyt
ls experiment_specs/debug2.yaml  # Should exist
python scripts/viz_simple.py --config-name debug2
```

### Issue: Still getting missing 'mean' error

**Check:** If using the updated script, it should work. If not:
```bash
# Verify you're using the updated viz_simple.py
grep "hydra_runner" scripts/viz_simple.py

# Should show the hydra_runner decorator
```

### Issue: Want command-line args instead of env vars

You can still pass Hydra overrides:
```bash
# Set custom batch size (overrides config)
python scripts/viz_simple.py --config-name debug2 \
    dataset.batch_size=16
```

## Output

Images are saved with the format:
```
output_dir/
├── b0000_i000_c42.jpg    # Batch 0, Image 0, Class 42
├── b0000_i001_c17.jpg
├── b0000_i002_c03.jpg
└── ...
```

Where:
- `b0000` = Batch index (0-padded to 4 digits)
- `i000` = Image index within batch (0-padded to 3 digits)  
- `c42` = Class label (if available)

## Summary

**Old way (broken):**
```bash
python scripts/viz_simple.py --config experiment_specs/debug2.yaml  # ❌ Missing mean/std
```

**New way (works):**
```bash
python scripts/viz_simple.py --config-name debug2  # ✅ Schema defaults merged
```

The key difference is using `hydra_runner` with `schema=ExperimentConfig`, which automatically merges your YAML config with all default values from the schema.


