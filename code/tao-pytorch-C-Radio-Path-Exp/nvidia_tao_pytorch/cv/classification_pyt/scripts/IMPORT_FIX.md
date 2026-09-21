# Import Error Fix

## Problem

You're getting this error:
```
ImportError: cannot import name 'GenericAlias' from 'types' 
(/tao-pt/nvidia_tao_pytorch/cv/classification_pyt/types/__init__.py)
```

### Root Cause

There's a local `types` module in `/nvidia_tao_pytorch/cv/classification_pyt/types/` that shadows Python's built-in `types` module. When code tries to import from the built-in `types`, Python finds the local one instead.

## Solutions

### Solution 1: Use the Simple Visualization Script (RECOMMENDED)

Use the minimal dependency version:

```bash
cd <EDGEAI_ROOT>/users/<user>/workspace/tmp/tao-pytorch-all/nvidia_tao_pytorch/cv/classification_pyt

python scripts/viz_simple.py \
    --config experiment_specs/debug2.yaml \
    --output-dir ./viz \
    --num-batches 3 \
    --images-per-batch 8
```

### Solution 2: Use the Standalone Version

```bash
python scripts/visualize_standalone.py \
    --config experiment_specs/debug2.yaml \
    --output-dir ./viz \
    --num-batches 3 \
    --images-per-batch 8 \
    --create-grid
```

### Solution 3: Python Script with Delayed Imports

Create your own script:

```python
# save_dataloader_images.py
import torch
import numpy as np
from PIL import Image
from pathlib import Path

def save_images(dataloader, output_dir, num_batches=3):
    """Save images from dataloader."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        # Extract images
        if isinstance(batch, dict):
            images = batch['img']
        else:
            images = batch[0]
        
        # Denormalize
        images = images * std + mean
        images = torch.clamp(images, 0, 1)
        
        # Save
        for i in range(min(8, images.shape[0])):
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)
            Image.fromarray(img).save(f"{output_dir}/batch{batch_idx}_img{i}.jpg")
        
        print(f"Saved batch {batch_idx}")

if __name__ == "__main__":
    # Import after the simple functions are defined
    from omegaconf import OmegaConf
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.pl_classification_data_module import CLDataModule
    
    config = OmegaConf.load("experiment_specs/debug2.yaml")
    dm = CLDataModule(config.dataset)
    dm.setup(stage="fit")
    
    save_images(dm.train_dataloader(), "./my_viz", num_batches=3)
```

Then run:
```bash
python save_dataloader_images.py
```

### Solution 4: Fix the types Module (Advanced)

If you want to permanently fix the issue, rename the local types module:

```bash
cd <EDGEAI_ROOT>/users/<user>/workspace/tmp/tao-pytorch-all/nvidia_tao_pytorch/cv/classification_pyt

# Rename the types module to avoid conflict
mv types custom_types

# Then update all imports in the codebase from:
# from nvidia_tao_pytorch.cv.classification_pyt.types import ...
# to:
# from nvidia_tao_pytorch.cv.classification_pyt.custom_types import ...
```

⚠️ **Warning**: This requires updating imports throughout the codebase.

## Quick Test

Test if the simple script works:

```bash
cd <EDGEAI_ROOT>/users/<user>/workspace/tmp/tao-pytorch-all/nvidia_tao_pytorch/cv/classification_pyt

python scripts/viz_simple.py \
    --config experiment_specs/debug2.yaml \
    --output-dir /tmp/test_viz \
    --num-batches 1 \
    --images-per-batch 4
```

Check the output:
```bash
ls -lh /tmp/test_viz/
```

## Scripts Available

| Script | Dependencies | Description |
|--------|--------------|-------------|
| `viz_simple.py` | Minimal | Simplest version, ~100 lines |
| `visualize_standalone.py` | Medium | More features, standalone functions |
| `visualize_augmentation.py` | Full | Full-featured with grid support |

## Troubleshooting

### Still getting import errors?

Try running Python with the `-I` flag to ignore user site packages:

```bash
python -I scripts/viz_simple.py --config experiment_specs/debug2.yaml --output-dir ./viz
```

### Module not found errors?

Make sure you're in the right directory:

```bash
cd <EDGEAI_ROOT>/users/<user>/workspace/tmp/tao-pytorch-all/nvidia_tao_pytorch/cv/classification_pyt
pwd  # Should show the classification_pyt directory
```

### Want to see the raw images without running scripts?

Use this minimal Python code:

```python
import torch
from PIL import Image

# Assuming you already have a batch
images = batch['img']  # Shape: (B, C, H, W)

# Quick denormalize and save
img = images[0]  # First image
img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
img = torch.clamp(img, 0, 1)
img = (img.permute(1, 2, 0).numpy() * 255).astype('uint8')
Image.fromarray(img).save('test.jpg')
```

## Summary

**Recommended approach:** Use `viz_simple.py` - it's the most reliable and avoids the import conflict.

```bash
python scripts/viz_simple.py \
    --config experiment_specs/debug2.yaml \
    --output-dir ./visualization \
    --num-batches 5 \
    --images-per-batch 8
```

Then check the output:
```bash
ls -lh ./visualization/
```

Images will be named: `b0000_i000_c42.jpg` (batch 0, image 0, class 42)

