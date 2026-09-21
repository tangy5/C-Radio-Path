#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone script to visualize dataloader - avoids import conflicts."""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image


def denormalize_tensor(tensor, mean, std):
    """
    Denormalize a tensor image.
    
    Args:
        tensor: Normalized tensor of shape (C, H, W) or (B, C, H, W)
        mean: List of mean values
        std: List of std values
    
    Returns:
        Denormalized tensor
    """
    mean_tensor = torch.tensor(mean).view(-1, 1, 1)
    std_tensor = torch.tensor(std).view(-1, 1, 1)
    
    if tensor.dim() == 4:
        mean_tensor = mean_tensor.unsqueeze(0)
        std_tensor = std_tensor.unsqueeze(0)
    
    mean_tensor = mean_tensor.to(tensor.device)
    std_tensor = std_tensor.to(tensor.device)
    
    # Denormalize: img = img * std + mean
    denorm = tensor * std_tensor + mean_tensor
    
    # Clip to [0, 1]
    return torch.clamp(denorm, 0, 1)


def tensor_to_image(tensor, mean, std):
    """
    Convert a tensor to PIL Image.
    
    Args:
        tensor: Tensor of shape (C, H, W)
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        PIL Image
    """
    # Denormalize
    tensor = denormalize_tensor(tensor, mean, std)
    
    # Move to CPU
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Convert from (C, H, W) to (H, W, C)
    np_img = tensor.permute(1, 2, 0).numpy()
    
    # Convert to uint8
    np_img = (np_img * 255).astype(np.uint8)
    
    return Image.fromarray(np_img)


def save_batch(batch, output_dir, batch_idx, num_images, mean, std, prefix="batch"):
    """
    Save a batch of images.
    
    Args:
        batch: Batch dict or tuple from dataloader
        output_dir: Directory to save images
        batch_idx: Batch index
        num_images: Number of images to save
        mean: Normalization mean
        std: Normalization std
        prefix: Filename prefix
    """
    # Handle different batch formats
    if isinstance(batch, dict):
        images = batch.get('img', batch.get('image'))
        labels = batch.get('class', batch.get('label'))
    elif isinstance(batch, (list, tuple)):
        images = batch[0]
        labels = batch[1] if len(batch) > 1 else None
    else:
        raise ValueError("Unsupported batch format")
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine number of images to save
    batch_size = images.shape[0]
    num_images = min(num_images, batch_size)
    
    # Save each image
    for i in range(num_images):
        img_tensor = images[i]
        
        # Convert to PIL Image
        pil_img = tensor_to_image(img_tensor, mean, std)
        
        # Create filename
        label_str = ""
        if labels is not None:
            if labels.dim() == 1:
                label = labels[i].item()
                label_str = f"_class{label}"
            else:
                label = labels[i].argmax().item()
                label_str = f"_class{label}"
        
        filename = f"{prefix}_batch{batch_idx:04d}_img{i:03d}{label_str}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        # Save image
        pil_img.save(filepath, quality=95)
    
    print(f"Saved {num_images} images from batch {batch_idx}")


def create_grid(batch, output_dir, batch_idx, mean, std, nrow=8):
    """Create a grid visualization of images."""
    import torchvision.utils as vutils
    
    # Handle different batch formats
    if isinstance(batch, dict):
        images = batch.get('img', batch.get('image'))
    elif isinstance(batch, (list, tuple)):
        images = batch[0]
    else:
        images = batch
    
    # Denormalize
    images = denormalize_tensor(images, mean, std)
    
    # Create grid
    grid = vutils.make_grid(images, nrow=nrow, padding=2, normalize=False)
    
    # Convert to PIL Image
    grid_img = tensor_to_image(grid, mean=[0, 0, 0], std=[1, 1, 1])  # Already denormalized
    
    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(output_dir, f"grid_batch{batch_idx:04d}.jpg")
    grid_img.save(filepath, quality=95)
    
    print(f"Saved grid for batch {batch_idx}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize augmented images from classification dataloader (standalone version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize training data
  python scripts/visualize_standalone.py \\
      --config experiment_specs/debug2.yaml \\
      --output-dir ./viz_train

  # Visualize with custom parameters
  python scripts/visualize_standalone.py \\
      --config experiment_specs/debug2.yaml \\
      --num-batches 3 \\
      --images-per-batch 8 \\
      --create-grid
        """
    )
    
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output-dir", type=str, default="./dataloader_viz", help="Output directory")
    parser.add_argument("--num-batches", type=int, default=5, help="Number of batches")
    parser.add_argument("--images-per-batch", type=int, default=8, help="Images per batch")
    parser.add_argument("--stage", type=str, default="fit", choices=["fit", "test"], help="Stage")
    parser.add_argument("--create-grid", action="store_true", help="Create grid visualizations")
    
    args = parser.parse_args()
    
    # Import here to avoid circular imports and types conflict
    print("Loading configuration...")
    from omegaconf import OmegaConf
    config = OmegaConf.load(args.config)
    
    # Get normalization parameters
    aug_config = config.dataset.get("augmentation", {})
    mean = list(aug_config.get("mean", [0.485, 0.456, 0.406]))
    std = list(aug_config.get("std", [0.229, 0.224, 0.225]))
    
    print(f"Using normalization: mean={mean}, std={std}")
    
    # Import data module after config is loaded
    print("Creating data module...")
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.pl_classification_data_module import CLDataModule
    
    data_module = CLDataModule(config.dataset)
    data_module.setup(stage=args.stage)
    
    # Get dataloader
    if args.stage == "fit":
        dataloader = data_module.train_dataloader()
        prefix = "train"
    else:
        dataloader = data_module.test_dataloader()
        prefix = "test"
    
    print(f"\nVisualizing {args.stage} dataloader...")
    print(f"  - Output directory: {args.output_dir}")
    print(f"  - Number of batches: {args.num_batches}")
    print(f"  - Images per batch: {args.images_per_batch}")
    print()
    
    # Visualize batches
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.num_batches:
            break
        
        # Save individual images
        save_batch(
            batch,
            output_dir=args.output_dir,
            batch_idx=batch_idx,
            num_images=args.images_per_batch,
            mean=mean,
            std=std,
            prefix=prefix
        )
        
        # Create grid if requested
        if args.create_grid:
            create_grid(
                batch,
                output_dir=args.output_dir,
                batch_idx=batch_idx,
                mean=mean,
                std=std,
                nrow=8
            )
    
    print(f"\n✅ Done! Images saved to {args.output_dir}")


if __name__ == "__main__":
    main()

