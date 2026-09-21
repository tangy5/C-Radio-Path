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

"""Reference parity dump entry-point.

Run (from TAO repo root, with PYTHONPATH pointing to reference project):
    cd ~/projects/tao-pytorch-all
    PYTHONPATH=~/projects/evfm:$PYTHONPATH \
      python -m nvidia_tao_pytorch.cv.classification_pyt.dataloader.parity.evfm \
        --output /tmp/parity/evfm.json
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import nvidia_tao_pytorch.cv.classification_pyt.dataloader.parity.evfm  # noqa: E402, F401
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.parity import dump_main  # noqa: E402

dump_main()
