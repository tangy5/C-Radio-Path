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

"""Simple script to visualize dataloader augmentations."""

import os
import sys
import argparse
from pathlib import Path

from omegaconf import OmegaConf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nvidia_tao_pytorch.cv.classification_pyt.dataloader.pl_classification_data_module import CLDataModule
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.visualize_dataloader import (
    visualize_classification_dataloader,
    DataLoaderVisualizer
)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize augmented images from classification dataloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize training data
  python visualize_augmentation.py --config experiment.yaml --output-dir ./viz_train
  
  # Visualize validation data
  python visualize_augmentation.py --config experiment.yaml --stage test --num-batches 3
  
  # Save more images per batch
  python visualize_augmentation.py --config experiment.yaml --images-per-batch 16
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./dataloader_visualization",
        help="Output directory for saved images (default: ./dataloader_visualization)"
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=5,
        help="Number of batches to visualize (default: 5)"
    )
    parser.add_argument(
        "--images-per-batch",
        type=int,
        default=8,
        help="Number of images to save per batch (default: 8)"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="fit",
        choices=["fit", "test", "predict"],
        help="Which dataloader to visualize: fit (train), test (val), or predict (default: fit)"
    )
    parser.add_argument(
        "--create-grid",
        action="store_true",
        help="Also create grid visualizations"
    )
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    config = OmegaConf.load(args.config)
    
    # Create data module
    print("Creating data module...")
    data_module = CLDataModule(config.dataset)
    
    # Visualize
    print(f"\nVisualizing {args.stage} dataloader...")
    print(f"  - Output directory: {args.output_dir}")
    print(f"  - Number of batches: {args.num_batches}")
    print(f"  - Images per batch: {args.images_per_batch}")
    print()
    
    visualize_classification_dataloader(
        data_module,
        output_dir=args.output_dir,
        num_batches=args.num_batches,
        num_images_per_batch=args.images_per_batch,
        stage=args.stage,
    )
    
    # Optionally create grids
    if args.create_grid:
        print("\nCreating grid visualizations...")
        data_module.setup(stage=args.stage)
        
        if args.stage == "fit":
            dataloader = data_module.train_dataloader()
        elif args.stage == "test":
            dataloader = data_module.test_dataloader()
        else:
            dataloader = data_module.predict_dataloader()
        
        aug_config = data_module.dataset_config.get("augmentation", {})
        mean = aug_config.get("mean", [0.485, 0.456, 0.406])
        std = aug_config.get("std", [0.229, 0.224, 0.225])
        
        visualizer = DataLoaderVisualizer(args.output_dir, mean=mean, std=std)
        
        for i, batch in enumerate(dataloader):
            if i >= 3:  # Create grids for first 3 batches
                break
            
            if isinstance(batch, (list, tuple)):
                batch = {'img': batch[0], 'class': batch[1] if len(batch) > 1 else None}
            
            visualizer.create_grid(
                batch,
                nrow=8,
                output_name=f"grid_batch{i:04d}.jpg"
            )
    
    print(f"\n✅ Done! Images saved to {args.output_dir}")


if __name__ == "__main__":
    main()

