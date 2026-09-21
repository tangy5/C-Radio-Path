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

"""
Color transformation matrices for RGB image augmentation.

Ported from EVFM for GPU-batched color augmentations. Provides 4x4 affine
matrices for contrast and hue/saturation that can be applied to batched
images in pure PyTorch (no adlr_ops_cpp dependency).

Reference: https://beesbuzz.biz/code/hsv_color_transforms.php
"""

import math

import torch


def contrast_matrix(contrast: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """
    Form a contrast transformation matrix for RGB images.

    The contrast matrix introduces a scaling around a center point.

    Args:
        contrast: tensor(float32) contrast value (scalar or vector). A value of 0
            keeps scaling untouched.
        center: tensor(float32) center value. For [0,1] range use 0.5.
            Scalar or vector.

    Returns:
        fp32 tensor (4, 4) if scalars, else (B, 4, 4) for vector inputs.
    """
    zero = torch.zeros_like(contrast)
    one = torch.ones_like(contrast)
    scale = one + contrast
    bias = -contrast * center

    # fmt: off
    m = torch.stack(
        [
            scale, zero,  zero,  zero,
            zero,  scale, zero,  zero,
            zero,  zero,  scale, zero,
            bias,  bias,  bias,  one,
        ],
        dim=-1,
    )
    # fmt: on
    shape = [-1, 4, 4] if len(contrast.shape) == 1 else [4, 4]
    return torch.reshape(m, shape)


# fmt: off
_HUE_SATURATION_CONST_MAT = torch.tensor(
    [
        [0.299, 0.299, 0.299, 0.0],
        [0.587, 0.587, 0.587, 0.0],
        [0.114, 0.114, 0.114, 0.0],
        [0.0,   0.0,   0.0,   1.0],
    ],
    dtype=torch.float32,
)

_HUE_SATURATION_SCH_MAT = torch.tensor(
    [
        [0.701,  -0.299, -0.300, 0.0],
        [-0.587,  0.413, -0.588, 0.0],
        [-0.114, -0.114,  0.886, 0.0],
        [0.0,     0.0,    0.0,   0.0],
    ],
    dtype=torch.float32,
)

_HUE_SATURATION_SSH_MAT = torch.tensor(
    [
        [0.168, -0.328,  1.25,  0.0],
        [0.330,  0.035, -1.05,  0.0],
        [-0.497, 0.292, -0.203, 0.0],
        [0.0,    0.0,    0.0,   0.0],
    ],
    dtype=torch.float32,
)
# fmt: on


def hue_saturation_matrix(hue: torch.Tensor, saturation: torch.Tensor) -> torch.Tensor:
    """
    Form a color saturation and hue transformation matrix for RGB images.

    Single matrix transform for both hue and saturation. Derived by transforming
    to HSV, modifying, and transforming back to RGB (linear approximation).

    Args:
        hue: Hue rotation in degrees (scalar or vector). 0 leaves hue unchanged.
        saturation: Saturation multiplier (scalar or vector). 1.0 leaves
            saturation unchanged. 0 removes all saturation.

    Returns:
        fp32 tensor (4, 4) if scalars, else (B, 4, 4) for vector inputs.
    """
    const_mat = _HUE_SATURATION_CONST_MAT.to(hue.device)
    sch_mat = _HUE_SATURATION_SCH_MAT.to(hue.device)
    ssh_mat = _HUE_SATURATION_SSH_MAT.to(hue.device)

    angle = hue * (math.pi / 180.0)
    sch = saturation * torch.cos(angle)
    ssh = saturation * torch.sin(angle)

    if len(hue.shape) == 1:
        batch_size = hue.shape[0]
        const_mat = torch.unsqueeze(const_mat, 0)
        sch_mat = torch.unsqueeze(sch_mat, 0)
        ssh_mat = torch.unsqueeze(ssh_mat, 0)
        sch = torch.reshape(sch, [batch_size, 1, 1])
        ssh = torch.reshape(ssh, [batch_size, 1, 1])

    return const_mat + sch * sch_mat + ssh * ssh_mat


def apply_color_transform(
    images: torch.Tensor,
    matrices: torch.Tensor,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
) -> torch.Tensor:
    """
    Apply 4x4 color transformation matrices to a batch of RGB images.

    Pure PyTorch implementation. Equivalent to adlr_ops_cpp.color_transform.

    Args:
        images: Batched images (B, C, H, W) in [0, 1] or [0, 255] range.
        matrices: Color transform matrices (B, 4, 4) or (4, 4).
        clamp_min: Minimum output value after transform.
        clamp_max: Maximum output value after transform.

    Returns:
        Transformed images (B, C, H, W), clamped.
    """
    if images.dim() != 4:
        raise ValueError("images must be (B, C, H, W)")

    B, C, H, W = images.shape
    if C != 3:
        raise ValueError("Expected 3-channel RGB images")

    # Flatten spatial dimensions: (B, 3, H, W) -> (B, 3, H*W)
    pixels = images.reshape(B, C, -1)

    # Homogeneous coords: (B, 4, H*W) with row 4 = 1
    ones = torch.ones(1, 1, H * W, dtype=images.dtype, device=images.device)
    pixels_h = torch.cat([pixels, ones.expand(B, 1, -1)], dim=1)

    # Apply matrix: (B, 4, 4) @ (B, 4, H*W) -> (B, 4, H*W)
    if matrices.dim() == 2:
        matrices = matrices.unsqueeze(0).expand(B, -1, -1)
    transformed = torch.bmm(matrices, pixels_h)

    # Drop homogeneous row, keep RGB: (B, 3, H*W)
    result = transformed[:, :3, :]

    # Reshape to image: (B, 3, H, W)
    result = result.reshape(B, C, H, W)

    return torch.clamp(result, clamp_min, clamp_max)
