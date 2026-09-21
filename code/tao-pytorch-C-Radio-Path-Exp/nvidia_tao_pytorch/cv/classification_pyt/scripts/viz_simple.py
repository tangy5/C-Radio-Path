#!/usr/bin/env python3
"""Simplest possible visualization script - minimal dependencies."""

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path


def save_images_from_dataloader(
    dataloader,
    output_dir,
    num_batches=5,
    images_per_batch=8,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
):
    """Save images from dataloader to disk."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)
    
    print(f"Saving images to {output_dir}...")
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        # Handle different batch formats
        if isinstance(batch, dict):
            images = batch.get('img', batch.get('image'))
            labels = batch.get('class', batch.get('label'))
        elif isinstance(batch, (list, tuple)):
            images = batch[0]
            labels = batch[1] if len(batch) > 1 else None
        else:
            images = batch
            labels = None
        
        # Denormalize: img = img * std + mean
        images = images * std_t.to(images.device) + mean_t.to(images.device)
        images = torch.clamp(images, 0, 1)
        
        # Save each image
        num_save = min(images_per_batch, images.shape[0])
        for i in range(num_save):
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            
            # Filename with optional label
            label_str = f"_c{labels[i].item()}" if labels is not None else ""
            filename = f"b{batch_idx:04d}_i{i:03d}{label_str}.jpg"
            pil_img.save(os.path.join(output_dir, filename), quality=95)
        
        print(f"  Batch {batch_idx}: saved {num_save} images")
    
    print(f"✅ Done! Check {output_dir}")


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Use hydra_runner to properly load config with schema defaults
from nvidia_tao_core.config.classification_pyt.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="experiment_spec",
    schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Visualize dataloader output."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.pl_classification_data_module import CLDataModule
    
    # Get visualization parameters from environment variables or use defaults
    output_dir = os.environ.get("VIZ_OUTPUT_DIR", "./dataloader_viz")
    num_batches = int(os.environ.get("VIZ_NUM_BATCHES", "3"))
    images_per_batch = int(os.environ.get("VIZ_IMAGES_PER_BATCH", "8"))
    
    # Get normalization from config (now properly merged with defaults)
    mean = list(cfg.dataset.augmentation.mean)
    std = list(cfg.dataset.augmentation.std)
    
    print(f"Output directory: {output_dir}")
    print(f"Mean: {mean}, Std: {std}")
    print(f"Visualizing {num_batches} batches, {images_per_batch} images per batch\n")
    
    # Create dataloader
    dm = CLDataModule(cfg.dataset)
    dm.setup(stage="fit")
    loader = dm.train_dataloader()
    
    # Visualize
    save_images_from_dataloader(
        loader,
        output_dir,
        num_batches=num_batches,
        images_per_batch=images_per_batch,
        mean=mean,
        std=std
    )


if __name__ == "__main__":
    main()

