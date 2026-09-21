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

import os
import tempfile
import torch

from nvidia_tao_core.config.centerpose.default_config import ExperimentConfig
from nvidia_tao_core.config.centerpose.model import CenterPoseModelConfig
from nvidia_tao_core.config.centerpose.dataset import CenterPoseDatasetConfig
from nvidia_tao_pytorch.cv.centerpose.model.centerpose import create_model
from nvidia_tao_pytorch.cv.centerpose.model.post_processing import HeatmapDecoder
from nvidia_tao_pytorch.cv.centerpose.model.centerpose import CenterPoseWrapped
from nvidia_tao_pytorch.cv.centerpose.utils.onnx_export import ONNXExporter


@pytest.fixture
def _test_experiment_spec():
    dataset_config = OmegaConf.structured(CenterPoseDatasetConfig())
    model_config = OmegaConf.structured(CenterPoseModelConfig())
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.dataset = dataset_config
    experiment_config.model = model_config
    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone", ["DLA34", "fan_small", "fan_base", "fan_large"])
@pytest.mark.parametrize("batch_size", [-1, 1])
def test_centerpose_onnx_export(_test_experiment_spec, backbone, batch_size):
    """Unit test for ONNX export on CenterPose model."""
    _test_experiment_spec["model"].backbone.model_type = backbone

    # Define input and output of ONNX
    if batch_size == -1:
        input_batch_size = 1
    else:
        input_batch_size = batch_size
    input_channel, input_height, input_width = 3, 512, 512
    input_names = ['inputs']
    output_names = ['bboxes', 'scores', 'kps', 'clses', 'obj_scale', 'kps_displacement_mean', 'kps_heatmap_mean']

    model_backbone = create_model(_test_experiment_spec["model"])
    model_backbone.eval()
    model_backbone.cuda()
    # Wrapped the heatmap decoder into the ONNX model to speed up the inference.
    hm_decoder = HeatmapDecoder()
    model = CenterPoseWrapped(model_backbone, hm_decoder)
    model.eval()
    model.cuda()

    dummy_input = torch.ones(input_batch_size, input_channel, input_height, input_width, device='cuda')
    os_handle, tmp_onnx_file = tempfile.mkstemp(suffix=".onnx")
    os.close(os_handle)

    onnx_export = ONNXExporter()
    onnx_export.export_model(model, batch_size, tmp_onnx_file, dummy_input,
                             input_names=input_names, output_names=output_names,
                             do_constant_folding=False)
    onnx_export.check_onnx(tmp_onnx_file)

    assert os.path.exists(tmp_onnx_file), "ONNX file was not generated properly!"
    os.remove(tmp_onnx_file)
