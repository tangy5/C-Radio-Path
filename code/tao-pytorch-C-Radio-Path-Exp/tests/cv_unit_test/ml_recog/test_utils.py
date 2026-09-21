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
import shutil
import pytest

from nvidia_tao_pytorch.cv.ml_recog.utils.common_utils import no_folders_in

TEST_DATA_DIR = "<SHARED_SCRATCH_MOUNT>/tao_ci/tao_pytorch/data/metric_learning_recognition"
TEST_OUTPUT_DIR = "tests/cv_unit_test/ml_recog/test_outputs"


@pytest.fixture
def _test_dir():
    output_dir = TEST_OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


def test_no_folders_in(_test_dir):
    assert not no_folders_in(os.path.join(TEST_DATA_DIR, "train"))
    assert no_folders_in(os.path.join(TEST_DATA_DIR, "train", "c000001"))
    shutil.rmtree(TEST_OUTPUT_DIR)
