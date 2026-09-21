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

"""Reference-side parity dump components.

These modules import from ``data.*`` which resolves when
PYTHONPATH includes the reference project root.
"""

import nvidia_tao_pytorch.cv.classification_pyt.dataloader.parity.evfm.misc  # noqa: F401
import nvidia_tao_pytorch.cv.classification_pyt.dataloader.parity.evfm.stages  # noqa: F401
import nvidia_tao_pytorch.cv.classification_pyt.dataloader.parity.evfm.transforms  # noqa: F401
import nvidia_tao_pytorch.cv.classification_pyt.dataloader.parity.evfm.spatial  # noqa: F401
import nvidia_tao_pytorch.cv.classification_pyt.dataloader.parity.evfm.collate  # noqa: F401
import nvidia_tao_pytorch.cv.classification_pyt.dataloader.parity.evfm.pipeline  # noqa: F401
