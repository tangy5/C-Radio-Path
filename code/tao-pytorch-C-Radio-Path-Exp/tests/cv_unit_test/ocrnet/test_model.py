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
import argparse
import torch
from nvidia_tao_pytorch.cv.ocrnet.model.model import Model, ExportModel

DEFAULT_WIDTH = 100
DEFAULT_HEIGHT = 32
DEFAULT_CHANNEL = 3 

@pytest.fixture
def _test_opt():
    parser = argparse.ArgumentParser()
    opt, _ = parser.parse_known_args()
    opt.input_channel = DEFAULT_CHANNEL
    opt.output_channel = 256
    opt.hidden_size = 256
    opt.imgW = DEFAULT_WIDTH
    opt.imgH = DEFAULT_HEIGHT
    opt.num_class = 37
    opt.num_fiducial = 20

    yield opt

@pytest.fixture
def _test_tensor():
    tensor = torch.randn(1, DEFAULT_CHANNEL, DEFAULT_HEIGHT, DEFAULT_WIDTH)
    
    yield tensor


@pytest.mark.ocrnet
@pytest.mark.parametrize("trans, feature, seq, pred, quantize",
                         [("TPS", "ResNet", "BiLSTM", "CTC", False),
                          ("TPS", "ResNet", "BiLSTM", "CTC", True),
                          ("None", "ResNet", "BiLSTM", "CTC", False),
                          ("None", "ResNet2X", "BiLSTM", "CTC", False),
                          ("TPS", "FAN_tiny_2X", "BiLSTM", "Attn", False)
                         ])
def test_model(_test_opt, trans, feature, seq, pred, quantize, _test_tensor):
    opt = _test_opt
    opt.Transformation = trans
    opt.FeatureExtraction = feature
    if opt.FeatureExtraction == "FAN_tiny_2X":
        opt.imgW = 200
        opt.imgH = 64
        opt.input_channel = 1
        opt.output_channel = 192
        _test_tensor = torch.randn(1, opt.input_channel, opt.imgH, opt.imgW)
    opt.SequenceModeling = seq 
    opt.Prediction = pred
    if opt.Prediction == "Attn":
        opt.batch_max_length = 25
    opt.quantize = quantize

    model = Model(opt)
    
    input_tensor = _test_tensor.cuda()
    model.train().cuda()
    with torch.no_grad():
        pred = model(input_tensor, None, is_train=False)
    
    model.eval()
    with torch.no_grad():
        pred = model(input_tensor, None, is_train=False)

    model.cpu()
    export_model = ExportModel(ocr_model = model, prediction_type=opt.SequenceModeling)