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
import torch
from torchvision.models import resnet18
from nvidia_tao_pytorch.cv.ocdnet.lr_schedulers.schedulers import WarmupPolyLR


@pytest.mark.cv_unit
def test_ocdnet_lr_scheduler():
    max_iter = 600 * 63
    model = resnet18()
    op = torch.optim.Adam(model.parameters(), 1e-3)
    sc = WarmupPolyLR(op, max_iters=max_iter, power=0.9, warmup_iters=3*63, warmup_method='constant', epochs=1)
    lr = []
    for i in range(max_iter):
        sc.step()
        #print(i, sc.last_epoch, sc.get_lr()[0])
        lr.append(sc.get_lr()[0])

