# Dataloader Visualization Guide

This guide explains how to visualize and save augmented images from the classification dataloader.

## Quick Start

### Method 1: Command Line Script

```bash
# Visualize training data (5 batches, 8 images each)
python scripts/visualize_augmentation.py \
    --config experiment_specs/debug2.yaml \
    --output-dir ./viz_train \
    --num-batches 5 \
    --images-per-batch 8

# Visualize validation data
python scripts/visualize_augmentation.py \
    --config experiment_specs/debug2.yaml \
    --stage test \
    --output-dir ./viz_val

# Create grid visualizations
python scripts/visualize_augmentation.py \
    --config experiment_specs/debug2.yaml \
    --create-grid
```

### Method 2: Python Code (Inline)

```python
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.pl_classification_data_module import CLDataModule
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.visualize_dataloader import DataLoaderVisualizer
from omegaconf import OmegaConf

# Load config
config = OmegaConf.load("experiment_specs/debug2.yaml")

# Create data module
data_module = CLDataModule(config.dataset)
data_module.setup(stage="fit")
train_loader = data_module.train_dataloader()

# Create visualizer
viz = DataLoaderVisualizer(
    output_dir="./my_viz",
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Visualize
viz.visualize_dataloader(train_loader, num_batches=5, num_images_per_batch=8)
```

### Method 3: Quick Inline Example

```python
# See scripts/quick_viz_example.py for complete examples
from scripts.quick_viz_example import quick_visualize

quick_visualize("experiment_specs/debug2.yaml", output_dir="./viz", num_batches=3)
```

## Features

### 1. Save Individual Images

```python
# Save images from a single batch
batch = next(iter(train_loader))
viz.save_batch(batch, batch_idx=0, num_images=8, prefix="train")
```

**Output:**
- `train_batch0000_img000_class42.jpg`
- `train_batch0000_img001_class17.jpg`
- etc.

### 2. Create Image Grids

```python
# Create a grid visualization
viz.create_grid(batch, nrow=8, output_name="batch_grid.jpg")
```

**Output:** Single image with multiple images arranged in a grid.

### 3. Denormalize Images

```python
# Denormalize a tensor back to [0, 1] range
normalized_tensor = batch['img'][0]  # Shape: (C, H, W)
denormalized = viz.denormalize(normalized_tensor)
```

### 4. Convert to PIL Images

```python
# Convert tensor to PIL Image for custom processing
pil_img = viz.tensor_to_image(batch['img'][0])
pil_img.save("my_image.jpg")
```

## Configuration

### Normalization Parameters

Make sure to use the correct mean and std values from your config:

```python
aug_config = config.dataset.augmentation
mean = aug_config.mean  # e.g., [0.485, 0.456, 0.406]
std = aug_config.std    # e.g., [0.229, 0.224, 0.225]

viz = DataLoaderVisualizer(output_dir="./viz", mean=mean, std=std)
```

### Common Normalization Values

- **ImageNet:** `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
- **Custom:** `mean=[0.5, 0.5, 0.5]`, `std=[0.5, 0.5, 0.5]`

## Output Directory Structure

```
output_dir/
├── train_batch0000_img000_class42.jpg
├── train_batch0000_img001_class17.jpg
├── train_batch0000_img002_class03.jpg
├── ...
├── train_batch0001_img000_class21.jpg
├── ...
└── grid_batch0000.jpg
```

## Advanced Usage

### Visualize Multiple Dataloaders

```python
# Training data
viz.visualize_dataloader(train_loader, num_batches=5, prefix="train")

# Validation data
data_module.setup(stage="test")
val_loader = data_module.val_dataloader()
viz.visualize_dataloader(val_loader, num_batches=3, prefix="val")
```

### Custom Batch Processing

```python
for batch_idx, batch in enumerate(train_loader):
    if batch_idx >= 10:
        break
    
    images = batch['img']
    labels = batch['class']
    
    # Custom filtering - only save images from class 5
    class_5_mask = labels == 5
    if class_5_mask.any():
        filtered_batch = {
            'img': images[class_5_mask],
            'class': labels[class_5_mask]
        }
        viz.save_batch(filtered_batch, batch_idx=batch_idx, prefix="class5")
```

### Compare Augmentations

```python
# Create two data modules with different augmentation configs
config1 = OmegaConf.load("config_strong_aug.yaml")
config2 = OmegaConf.load("config_weak_aug.yaml")

viz1 = DataLoaderVisualizer("./viz_strong_aug")
viz2 = DataLoaderVisualizer("./viz_weak_aug")

data_module1 = CLDataModule(config1.dataset)
data_module2 = CLDataModule(config2.dataset)

data_module1.setup(stage="fit")
data_module2.setup(stage="fit")

viz1.visualize_dataloader(data_module1.train_dataloader(), num_batches=5)
viz2.visualize_dataloader(data_module2.train_dataloader(), num_batches=5)
```

### Side-by-Side Comparison

```python
import matplotlib.pyplot as plt
from PIL import Image

# Load images
img1 = Image.open("./viz_strong_aug/train_batch0000_img000.jpg")
img2 = Image.open("./viz_weak_aug/train_batch0000_img000.jpg")

# Plot side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(img1)
axes[0].set_title("Strong Augmentation")
axes[0].axis('off')

axes[1].imshow(img2)
axes[1].set_title("Weak Augmentation")
axes[1].axis('off')

plt.tight_layout()
plt.savefig("augmentation_comparison.png", dpi=150, bbox_inches='tight')
```

## Troubleshooting

### Issue: Images look wrong/too bright/too dark

**Solution:** Check your normalization parameters match your config:
```python
# Print actual values being used
print(f"Mean: {viz.mean}")
print(f"Std: {viz.std}")

# Check config
print(f"Config mean: {config.dataset.augmentation.mean}")
print(f"Config std: {config.dataset.augmentation.std}")
```

### Issue: WebDataset vs CLDataset format differences

The visualizer handles both formats automatically:
```python
# For CLDataset: batch is a dict with 'img' and 'class'
# For WebDataset: batch might be a tuple (images, labels)

# Visualizer converts automatically, but if needed:
if isinstance(batch, (list, tuple)):
    batch = {'img': batch[0], 'class': batch[1] if len(batch) > 1 else None}
```

### Issue: Out of memory

**Solution:** Reduce batch size or number of images:
```python
viz.visualize_dataloader(
    train_loader,
    num_batches=2,  # Fewer batches
    num_images_per_batch=4  # Fewer images per batch
)
```

## Integration with Training Code

Add visualization to your training script:

```python
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.visualize_dataloader import DataLoaderVisualizer

# In your training setup
if args.visualize_data:
    viz = DataLoaderVisualizer(
        output_dir=os.path.join(args.results_dir, "dataloader_viz"),
        mean=config.dataset.augmentation.mean,
        std=config.dataset.augmentation.std
    )
    
    # Visualize before training starts
    viz.visualize_dataloader(
        train_loader,
        num_batches=3,
        num_images_per_batch=8,
        prefix="epoch0"
    )
```

## API Reference

### `DataLoaderVisualizer`

```python
DataLoaderVisualizer(
    output_dir: str,                    # Directory to save images
    mean: List[float] = None,           # Normalization mean
    std: List[float] = None,            # Normalization std
)
```

**Methods:**
- `denormalize(tensor)` - Denormalize a tensor
- `tensor_to_image(tensor)` - Convert tensor to PIL Image
- `save_batch(batch, batch_idx, num_images, prefix)` - Save images from a batch
- `visualize_dataloader(dataloader, num_batches, num_images_per_batch, prefix)` - Visualize entire dataloader
- `create_grid(batch, nrow, output_name)` - Create grid visualization

### `visualize_classification_dataloader`

```python
visualize_classification_dataloader(
    data_module: CLDataModule,          # Lightning data module
    output_dir: str,                    # Output directory
    num_batches: int = 5,               # Number of batches
    num_images_per_batch: int = 8,      # Images per batch
    stage: str = "fit",                 # 'fit', 'test', or 'predict'
)
```

## See Also

- `visualize_dataloader.py` - Main visualization module
- `visualize_augmentation.py` - Command-line script
- `quick_viz_example.py` - Inline usage examples

