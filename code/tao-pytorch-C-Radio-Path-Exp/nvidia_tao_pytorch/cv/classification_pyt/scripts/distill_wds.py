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

"""Distill classification model with WebDataset support (no schema validation)."""

import os
from pytorch_lightning import LightningModule
from omegaconf import OmegaConf

from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner

from nvidia_tao_pytorch.cv.classification_pyt.distillation.distiller import (
    ClassDistiller,
)

from nvidia_tao_pytorch.cv.classification_pyt.scripts.train import run_experiment

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

    run_experiment(
        experiment_config=cfg, key=cfg.encryption_key, lightning_module=ClassDistiller
    )


if __name__ == "__main__":
    main()
