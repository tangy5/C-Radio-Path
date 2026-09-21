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
from nvidia_tao_core.config.depth_net.default_config import ExperimentConfig, DepthNetDatasetConfig, DepthNetModelConfig
from nvidia_tao_pytorch.cv.depth_net.model.build_pl_model import build_pl_model

@pytest.fixture
def _test_experiment_spec():
    dataset_config = OmegaConf.structured(DepthNetDatasetConfig())
    model_config = OmegaConf.structured(DepthNetModelConfig())
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.dataset = dataset_config
    experiment_config.model = model_config
    yield experiment_config

@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.model
@pytest.mark.parametrize("model_type", ["RelativeDepthAnything", "MetricDepthAnything"])
def test_depth_net_model_build(_test_experiment_spec, model_type):
    """Tests if the DepthNetPlModel can be instantiated."""
    _test_experiment_spec.model.model_type = model_type
    try:
        model = build_pl_model(_test_experiment_spec)
        assert model is not None, "Depth Model instantiation failed."
    except Exception as e:
        pytest.fail(f"Depth Model instantiation raised an exception: {e}")