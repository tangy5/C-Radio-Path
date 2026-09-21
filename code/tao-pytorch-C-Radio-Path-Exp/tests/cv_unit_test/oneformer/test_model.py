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

import pytest
from omegaconf import OmegaConf

from nvidia_tao_core.config.oneformer.model import OneFormerModelConfig, Swin
from nvidia_tao_core.config.oneformer.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.oneformer.model.oneformer_model import OneFormerModel


@pytest.fixture
def _test_experiment_spec():
    swin_config = OmegaConf.structured(Swin())
    model_config = OmegaConf.structured(OneFormerModelConfig())
    model_config.backbone.swin = swin_config
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.model = model_config
    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone", ["swin"])
@pytest.mark.parametrize("name", ['tiny', 'large'])
@pytest.mark.parametrize("export", [False, True])
def test_oneformer_model(_test_experiment_spec, backbone, name, export):
    _test_experiment_spec["model"].backbone.name = "D2SwinTransformer"
    # Configure the Swin variant by setting the appropriate parameters
    if name == 'tiny':
        _test_experiment_spec["model"].backbone.swin.embed_dim = 96
        _test_experiment_spec["model"].backbone.swin.depths = [2, 2, 6, 2]
        _test_experiment_spec["model"].backbone.swin.num_heads = [3, 6, 12, 24]
    elif name == 'large':
        _test_experiment_spec["model"].backbone.swin.embed_dim = 192
        _test_experiment_spec["model"].backbone.swin.depths = [2, 2, 18, 2]
        _test_experiment_spec["model"].backbone.swin.num_heads = [6, 12, 24, 48]

    model = OneFormerModel(_test_experiment_spec)
    assert model is not None
