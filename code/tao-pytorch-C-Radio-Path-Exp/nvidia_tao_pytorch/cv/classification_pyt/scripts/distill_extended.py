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

"""Distill classification model with Extended Schema supporting WebDataset."""

import os
from pytorch_lightning import LightningModule

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner

from nvidia_tao_pytorch.cv.classification_pyt.distillation.distiller import (
    ClassDistiller,
)

from nvidia_tao_pytorch.cv.classification_pyt.scripts.train import run_experiment

# Import extended config
from nvidia_tao_pytorch.cv.classification_pyt.config.extended_config import (
    ExtendedExperimentConfig,
)

spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification with EXTENDED schema supporting webdataset
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="distill",
    schema=ExtendedExperimentConfig,  # Use extended schema with tar_data_sources
)
@monitor_status(name="Class_pt", mode="distill")
def main(cfg: ExtendedExperimentConfig) -> None:
    """Run the distillation process."""
    # This is for resuming to work without needing to save the teacher weights
    LightningModule.strict_loading = False

    run_experiment(
        experiment_config=cfg, key=cfg.encryption_key, lightning_module=ClassDistiller
    )


if __name__ == "__main__":
    main()
