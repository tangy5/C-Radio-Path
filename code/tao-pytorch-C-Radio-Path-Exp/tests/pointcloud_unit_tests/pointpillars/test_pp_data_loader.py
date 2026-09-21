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

import os
import numpy as np
import pytest

from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.datasets.processor.data_processor import (
    VoxelGeneratorWrapper
)

tmp_top_dir = "<SHARED_SCRATCH_MOUNT>/tao_ci/tao_pytorch/data"

@pytest.mark.pointcloud_unit
def test_voxel_generator():
    """Test VoxelGenerator."""
    voxel_generator = VoxelGeneratorWrapper(
        vsize_xyz=[0.2, 0.2, 5.8],
        coors_range_xyz=[-51.2, -51.2, -1.4, 51.2, 51.2, 4.4],
        num_point_features=4,
        max_num_points_per_voxel=32,
        max_num_voxels=16000
    )
    points = np.fromfile(
        os.path.join(tmp_top_dir, "pointcloud", "102.bin"),
        dtype=np.float32
    ).reshape(-1, 4)
    voxels = voxel_generator.generate(points)[0]
    assert voxels.shape[1:] == (32, 4)
    assert voxels.shape[0] <= 16000
