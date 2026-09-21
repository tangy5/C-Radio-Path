# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nvidia_tao_core.config.mask2former.model import Mask2FormerModelConfig, Swin, Backbone
from nvidia_tao_core.config.mask2former.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.mask2former.model.mask2former import MaskFormerModel


@pytest.fixture
def _test_experiment_spec():
    swin_config = OmegaConf.structured(Swin())
    bb_config = OmegaConf.structured(Backbone())
    model_config = OmegaConf.structured(Mask2FormerModelConfig())
    model_config.backbone.swin = swin_config
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.model = model_config
    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone", ["swin"])
@pytest.mark.parametrize("name", ['tiny', 'large'])
@pytest.mark.parametrize("export", [False, True])
def test_mask2former_model(_test_experiment_spec, backbone, name, export):
    _test_experiment_spec["model"].backbone.type = backbone
    _test_experiment_spec["model"].backbone.swin.type = name

    model = MaskFormerModel(_test_experiment_spec)
