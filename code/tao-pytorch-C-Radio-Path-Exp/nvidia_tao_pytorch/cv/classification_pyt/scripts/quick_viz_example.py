#!/usr/bin/env python3
"""Quick inline example for visualizing dataloader output."""

import torch
from omegaconf import OmegaConf
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nvidia_tao_pytorch.cv.classification_pyt.dataloader.pl_classification_data_module import CLDataModule
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.visualize_dataloader import DataLoaderVisualizer


def quick_visualize(config_path: str, output_dir: str = "./quick_viz", num_batches: int = 3):
    """
    Quick function to visualize dataloader output.
    
    Args:
        config_path: Path to experiment YAML config
        output_dir: Where to save images
        num_batches: Number of batches to save
    """
    # Load config
    config = OmegaConf.load(config_path)
    
    # Create data module
    data_module = CLDataModule(config.dataset)
    data_module.setup(stage="fit")
    
    # Get dataloader
    train_loader = data_module.train_dataloader()
    
    # Get normalization parameters
    aug_config = config.dataset.get("augmentation", {})
    mean = aug_config.get("mean", [0.485, 0.456, 0.406])
    std = aug_config.get("std", [0.229, 0.224, 0.225])
    
    # Create visualizer
    viz = DataLoaderVisualizer(output_dir, mean=mean, std=std)
    
    # Save images
    print(f"Saving {num_batches} batches to {output_dir}...")
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
        
        # Handle different batch formats
        if isinstance(batch, (list, tuple)):
            batch = {'img': batch[0], 'class': batch[1] if len(batch) > 1 else None}
        
        # Save individual images
        viz.save_batch(batch, batch_idx=batch_idx, num_images=8)
        
        # Save grid
        viz.create_grid(batch, nrow=4, output_name=f"grid_batch{batch_idx}.jpg")
    
    print(f"✅ Done! Check {output_dir}")


# ============================================================================
# Example 1: Direct usage in Python
# ============================================================================

if __name__ == "__main__":
    # Example usage
    config_path = "/path/to/your/experiment.yaml"
    
    # Visualize training data
    quick_visualize(config_path, output_dir="./viz_train", num_batches=5)
    
    
    # ========================================================================
    # Example 2: More detailed control
    # ========================================================================
    
    # Load config
    config = OmegaConf.load(config_path)
    
    # Create data module
    data_module = CLDataModule(config.dataset)
    data_module.setup(stage="fit")
    
    # Get a single batch
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    
    # Handle batch format
    if isinstance(batch, (list, tuple)):
        images = batch[0]
        labels = batch[1] if len(batch) > 1 else None
    else:
        images = batch['img']
        labels = batch.get('class', None)
    
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape if labels is not None else 'None'}")
    print(f"Image range: [{images.min().item():.3f}, {images.max().item():.3f}]")
    
    # Create visualizer
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    viz = DataLoaderVisualizer("./custom_viz", mean=mean, std=std)
    
    # Save first 4 images
    viz.save_batch({'img': images, 'class': labels}, batch_idx=0, num_images=4)
    
    # Create a grid
    viz.create_grid({'img': images}, nrow=8, output_name="training_batch_grid.jpg")
    
    
    # ========================================================================
    # Example 3: Manual denormalization and saving
    # ========================================================================
    
    import torchvision.utils as vutils
    from PIL import Image
    import numpy as np
    
    # Get a batch
    batch = next(iter(train_loader))
    if isinstance(batch, (list, tuple)):
        images = batch[0]
    else:
        images = batch['img']
    
    # Manual denormalization
    mean_tensor = torch.tensor(mean).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std).view(1, 3, 1, 1)
    denorm_images = images * std_tensor + mean_tensor
    denorm_images = torch.clamp(denorm_images, 0, 1)
    
    # Save individual images
    for i in range(min(4, denorm_images.shape[0])):
        img = denorm_images[i]
        
        # Convert to PIL
        if img.is_cuda:
            img = img.cpu()
        np_img = img.permute(1, 2, 0).numpy()
        np_img = (np_img * 255).astype(np.uint8)
        pil_img = Image.fromarray(np_img)
        
        # Save
        pil_img.save(f"manual_save_{i}.jpg", quality=95)
    
    # Create grid using torchvision
    grid = vutils.make_grid(denorm_images[:16], nrow=4, padding=2)
    grid_np = grid.permute(1, 2, 0).numpy()
    grid_np = (grid_np * 255).astype(np.uint8)
    grid_img = Image.fromarray(grid_np)
    grid_img.save("manual_grid.jpg", quality=95)
    
    print("✅ Manual save complete!")

