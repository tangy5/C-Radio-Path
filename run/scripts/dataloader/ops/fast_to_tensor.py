# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Fast PIL Image to Tensor conversion.

Converts an RGB PIL Image (HWC uint8) to a CHW float32 tensor in [0, 1].
Uses a C++ SSE4/AVX2 SIMD kernel on x86_64 for ~4x throughput; falls back
to a pure-Python path when the C++ extension is not available or when
running on a non-x86 architecture.
"""

import logging

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_cpp_fast_to_tensor = None
try:
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.ops.FastToTensor import (
        fast_to_tensor_cpu,
        is_available,
    )
    if is_available():
        _cpp_fast_to_tensor = fast_to_tensor_cpu
    else:
        logger.info("FastToTensor C++ extension built but SIMD not available on this architecture; using Python fallback.")
except ImportError:
    logger.info("FastToTensor C++ extension not built; using Python fallback.")


def fast_to_tensor_fallback(pic: Image.Image) -> torch.Tensor:
    """Pure-Python HWC uint8 PIL Image to CHW float32 tensor in [0, 1]."""
    np_img = np.array(pic, copy=False)
    img = torch.from_numpy(np_img).permute(2, 0, 1)
    fp_img = img.to(dtype=torch.float32, memory_format=torch.contiguous_format)
    fp_img.div_(255)
    return fp_img


def fast_to_tensor(pic: Image.Image) -> torch.Tensor:
    """Convert a PIL Image (RGB, HWC uint8) to a CHW float32 tensor in [0, 1].

    Uses the C++ SIMD kernel if available, otherwise falls back to
    ``fast_to_tensor_fallback`` (pure Python).
    """
    if not isinstance(pic, Image.Image):
        raise TypeError(f"Expected PIL Image, got {type(pic)}")

    if _cpp_fast_to_tensor is not None:
        np_img = np.array(pic, copy=False)
        return _cpp_fast_to_tensor(torch.from_numpy(np_img))

    return fast_to_tensor_fallback(pic)
