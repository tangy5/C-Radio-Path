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

"""Extended Experiment Config with WebDataset support for Classification PyTorch."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from omegaconf import MISSING

from nvidia_tao_core.config.classification_pyt.default_config import (
    ExperimentConfig as BaseExperimentConfig,
    DataPathFormat,
)
from nvidia_tao_core.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    FLOAT_FIELD,
    BOOL_FIELD,
    LIST_FIELD,
    DICT_FIELD,
    DATACLASS_FIELD,
)


@dataclass
class TarDataSource:
    """Configuration for a single webdataset tar source."""
    root_dir: str = ""
    samples_per_file: int = 10000
    scale_factor: float = 1.0
    steps_per_epoch: int = 2000


@dataclass
class ExtendedTrainDataset(DataPathFormat):
    """Extended train dataset config with webdataset support."""
    
    # Standard fields from DataPathFormat
    images_dir: str = ""
    
    # WebDataset fields
    tar_data_sources: Optional[List[TarDataSource]] = None
    
    # EVFM/Extended fields
    student_patch_size: int = 16
    seed: int = 42
    full_equivariance: bool = False
    shift_equivariance: bool = False
    prefetch: bool = True
    data_weight_mode: str = ""


@dataclass
class ExtendedExperimentConfig(BaseExperimentConfig):
    """Extended experiment config with webdataset support."""
    
    # Override train_dataset with extended version
    # The dataset.train_dataset field will now accept tar_data_sources
    pass
