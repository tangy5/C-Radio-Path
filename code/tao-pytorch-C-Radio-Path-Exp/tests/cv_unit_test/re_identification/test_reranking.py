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
import numpy as np
from nvidia_tao_pytorch.cv.re_identification.utils.re_ranking import re_rank, rerank_gpu

@pytest.mark.cv_unit
def test_reranking():
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Set a fixed seed for reproducibility
    torch.manual_seed(0)  # Set the seed for CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)  # Set the seed for all GPUs

    # Set defaults for reranking
    feature_size = 256  # Feature size
    k1 = 20
    k2 = 6
    lambda_value = 0.3

    # Create dummy tensors for probFea and galFea
    probFea = torch.randn(10, feature_size, dtype=torch.float16, device=device)
    probFea = torch.nn.functional.normalize(probFea, dim=1, p=2)
    galFea = torch.randn(100, feature_size, dtype=torch.float16, device=device)
    galFea = torch.nn.functional.normalize(galFea, dim=1, p=2)
    
    dist_matrix_1 = rerank_gpu(probFea, galFea, k1, k2, lambda_value)
    dist_matrix_1 = dist_matrix_1.cpu().numpy()
    probFea = probFea.cpu().numpy()
    galFea = galFea.cpu().numpy()
    
    # Perform re-ranking on gpu
    dist_matrix_2 = re_rank(probFea, galFea, k1, k2, lambda_value)

    # Compare the results
    assert np.isclose(dist_matrix_1[0][0], dist_matrix_2[0][0], atol=0.1)
