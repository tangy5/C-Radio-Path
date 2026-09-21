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

import pytest
from omegaconf import OmegaConf

from nvidia_tao_core.config.mal.default_config import (
    ExperimentConfig,
    MALModelConfig
)
from nvidia_tao_pytorch.cv.mal.models.vit_builder import build_model


@pytest.fixture
def _test_cfg():
    model_config = OmegaConf.structured(MALModelConfig())
    cfg = OmegaConf.structured(ExperimentConfig())
    cfg.model = model_config
    yield cfg


@pytest.mark.cv_unit
@pytest.mark.parametrize("arch",
                         ["vit-mae-base/16",
                          "vit-mae-large/16",
                          "fan_tiny_12_p16_224",
                          "fan_small_12_p16_224",
                          "fan_base_18_p16_224",
                          "fan_large_24_p16_224",
                          "fan_tiny_8_p4_hybrid",
                          "fan_small_12_p4_hybrid",
                          "fan_base_16_p4_hybrid",
                          "fan_large_16_p4_hybrid"])
def test_mal_backbone(_test_cfg, arch):
    _test_cfg.model.arch = arch
    _test_cfg.model.frozen_stages = [0, -1]
    build_model(_test_cfg)
