#!/usr/bin/env python
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""Distill classification model with WebDataset support, Fast Validation, and DDP.

This is a modified version that:
- Works without schema validation to support tar_data_sources
- Supports limit_val_batches for fast validation (2 iterations)
- Uses DistributedSampler for validation to properly shard data across GPUs
- Optimized for multi-GPU distillation
"""

import os
import math
from pytorch_lightning import LightningModule
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.classification_pyt.distillation.distiller import ClassDistiller
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
from pytorch_lightning import Trainer

spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def update_results_dir_wds(cfg, task="distill"):
    """Update global results_dir based on task.results_dir.
    
    This is a modified version that works without schema validation.
    """
    # Allow modifications to the config structure
    OmegaConf.set_struct(cfg, False)
    
    # Ensure distill section exists
    if "distill" not in cfg:
        cfg["distill"] = {}
    
    # Set results_dir
    if cfg.get("results_dir"):
        cfg["results_dir"] = os.path.join(cfg["results_dir"], task)
        cfg["distill"]["results_dir"] = cfg["results_dir"]
    else:
        raise ValueError("You need to set results_dir in the config")
    
    print(f"{task.capitalize()} results will be saved at: {cfg['results_dir']}")
    return cfg


def is_dist_avail_and_initialized():
    """Check if DDP is initialized."""
    if not dist.is_available():
        return False
    return dist.is_initialized()


def get_world_size():
    """Get world size."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """Get global rank."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


import pytorch_lightning as pl

class DistributedCLDataModule(pl.LightningDataModule):
    """Wrapper to add distributed validation sampler."""
    
    def __init__(self, base_dm):
        super().__init__()
        self.base_dm = base_dm
        self.dataset_config = base_dm.dataset_config
        self.batch_size = base_dm.batch_size
        self.num_workers = base_dm.num_workers
        
    def __getattr__(self, name):
        """Forward attribute access to base data module."""
        if name in ['base_dm', 'dataset_config', 'batch_size', 'num_workers']:
            return object.__getattribute__(self, name)
        return getattr(self.base_dm, name)
    
    def train_dataloader(self):
        """Forward to base data module."""
        return self.base_dm.train_dataloader()
    
    def val_dataloader(self):
        """Build the dataloader for validation with DistributedSampler.
        
        This ensures each GPU processes a different portion of validation data.
        """
        world_size = get_world_size()
        rank = get_rank()
        
        # Get base validation loader
        base_loader = self.base_dm.val_dataloader()
        val_dataset = base_loader.dataset
        
        # Create DistributedSampler for validation
        # Each GPU gets a different subset of validation data
        if world_size > 1:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,  # No shuffle for validation
                drop_last=False
            )
            
            val_loader = DataLoader(
                val_dataset,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                sampler=val_sampler,
                collate_fn=val_dataset.collate_fn,
                pin_memory=True,
            )
            
            if rank == 0:
                total_samples = len(val_dataset)
                samples_per_gpu = math.ceil(total_samples / world_size)
                print(f"[DDP Validation] Total samples: {total_samples}")
                print(f"[DDP Validation] GPUs: {world_size}")
                print(f"[DDP Validation] Samples per GPU: ~{samples_per_gpu}")
            
            return val_loader
        else:
            # Single GPU - use original loader
            return base_loader


def run_experiment_fastval_ddp(experiment_config, key, lightning_module=ClassDistiller):
    """Start the training with fast validation and DDP support."""
    resume_ckpt, trainer_kwargs = initialize_train_experiment(experiment_config, key)

    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.pl_classification_data_module import CLDataModule
    
    # Create base data module
    base_dm = CLDataModule(experiment_config.dataset, experiment_config=experiment_config)
    base_dm.setup(stage="fit")
    
    # Wrap with distributed validation support
    dm = DistributedCLDataModule(base_dm)
    
    model = lightning_module(experiment_config)

    num_nodes = experiment_config.train.num_nodes
    clip_grad_norm = experiment_config.train.clip_grad_norm

    if experiment_config.train.precision.lower() == 'fp16':
        precision = '16-mixed'
    elif experiment_config.train.precision.lower() == 'bf16':
        precision = 'bf16-mixed'
    elif experiment_config.train.precision.lower() == 'fp32':
        precision = '32-true'
    else:
        raise NotImplementedError(
            f"{experiment_config.train.precision} is not supported. Only bf16, fp16, and fp32 are supported")

    strategy = 'auto'
    if len(trainer_kwargs['devices']) > 1:
        strategy = 'ddp_find_unused_parameters_true'

    # Fast validation support: limit_val_batches from config
    fastval_overrides = {}
    
    # Check if limit_val_batches is set in config
    if hasattr(experiment_config.train, 'limit_val_batches'):
        limit_val_batches = experiment_config.train.limit_val_batches
        if limit_val_batches is not None:
            fastval_overrides["limit_val_batches"] = limit_val_batches
            if get_rank() == 0:
                print(f"[Fast Validation] Limiting validation to {limit_val_batches} batches per GPU")
                total_val_samples = limit_val_batches * experiment_config.dataset.batch_size * get_world_size()
                print(f"[Fast Validation] Total validation samples: ~{total_val_samples}")

    # Quick test support: max_steps from config (via Hydra override +trainer.max_steps=N)
    quick_test_overrides = {}
    if hasattr(experiment_config, 'trainer') and hasattr(experiment_config.trainer, 'max_steps'):
        max_steps = experiment_config.trainer.max_steps
        if max_steps and max_steps > 0:
            quick_test_overrides["max_steps"] = max_steps
            if get_rank() == 0:
                print(f"[Quick Test] Limiting training to max_steps={max_steps}")

    trainer = Trainer(
        **trainer_kwargs,
        gradient_clip_val=clip_grad_norm,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=precision,
        use_distributed_sampler=False,  # We handle it manually for validation
        sync_batchnorm=True,
        **fastval_overrides,
        **quick_test_overrides,
    )

    trainer.fit(model, dm, ckpt_path=resume_ckpt)


# Load experiment specification WITHOUT schema validation to support tar_data_sources
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="distill",
)
def main(cfg) -> None:
    """Run the distillation process."""
    # This is for resuming to work without needing to save the teacher weights
    LightningModule.strict_loading = False
    
    # Update results_dir manually
    cfg = update_results_dir_wds(cfg, task="distill")
    
    # Create results directory
    os.makedirs(cfg["results_dir"], exist_ok=True)
    
    # Save config
    OmegaConf.save(cfg, os.path.join(cfg["results_dir"], "experiment.yaml"))

    run_experiment_fastval_ddp(
        experiment_config=cfg, key=cfg.encryption_key, lightning_module=ClassDistiller
    )


if __name__ == "__main__":
    main()
