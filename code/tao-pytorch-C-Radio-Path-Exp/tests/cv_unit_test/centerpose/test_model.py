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

from nvidia_tao_core.config.centerpose.default_config import ExperimentConfig
from nvidia_tao_core.config.centerpose.dataset import CenterPoseDatasetConfig
from nvidia_tao_core.config.centerpose.model import CenterPoseModelConfig

from nvidia_tao_pytorch.cv.centerpose.model.centerpose import create_model


@pytest.fixture
def _test_experiment_spec():
    dataset_config = OmegaConf.structured(CenterPoseDatasetConfig())
    model_config = OmegaConf.structured(CenterPoseModelConfig())
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.dataset = dataset_config
    experiment_config.model = model_config
    yield experiment_config

@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone, use_pretrained", 
                         [("DLA34", False),
                          # TODO @vpraveen/@jianhey: enable this test once torchhub
                          # connection to torch hub is restored.
                          # ("DLA34", True),  
                          ("fan_small", False),
                          ("fan_base", False),
                          ("fan_large", False)])
@pytest.mark.parametrize("down_ratio", [2, 4, 8, 16])
def test_centerpose_model(_test_experiment_spec, backbone, use_pretrained, down_ratio):
    _test_experiment_spec["model"].backbone.model_type = backbone
    _test_experiment_spec["model"].use_pretrained = use_pretrained
    _test_experiment_spec["model"].down_ratio = down_ratio

    create_model(_test_experiment_spec["model"])
