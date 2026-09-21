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

"""Four-corner bounds tracking through spatial transforms.

``Quad`` tracks the four corners of an image as 2D vertices.  Each
spatial transform (translate, scale, rotate, flip, arbitrary STM) is
applied to the vertices so that downstream code (e.g. ``equivariant_collate``)
can determine which region of the output canvas contains valid pixels.
"""

import torch


class Quad:
    """Axis-aligned quadrilateral that tracks image bounds.

    Initialised from an image tensor's spatial dimensions.  Methods
    mirror those on ``DeferImage`` so both can be transformed in lockstep.

    Attributes:
        bounds: ``(4, 2)`` float32 tensor of corner coordinates
            ``[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]``.
    """

    def __init__(self, image: torch.Tensor):
        self.bounds = torch.tensor([
            [0, 0],
            [image.shape[-1], 0],
            [image.shape[-1], image.shape[-2]],
            [0, image.shape[-2]],
        ], dtype=torch.float32)

    def apply_stm(self, stm: torch.Tensor, **kwargs):
        """Apply a 3x3 homogeneous transformation matrix to the bounds."""
        self.bounds = _apply_single_stm(self.bounds, stm)

    def translate(self, delta_vector: torch.Tensor):
        self.bounds += delta_vector

    def scale(self, scale_vector: torch.Tensor, **kwargs):
        self.bounds *= scale_vector

    def rotate(self, rot_mat: torch.Tensor):
        self.bounds = self.bounds @ rot_mat.T

    def flip(self, x: float):
        self.bounds[:, 0] -= x
        self.bounds[:, 0] *= -1
        self.bounds[:, 0] += x


def _apply_single_stm(vertices: torch.Tensor, stm: torch.Tensor):
    """Apply a 3x3 homogeneous transformation to 2D vertices.

    Args:
        vertices: ``(N, 2)`` tensor of 2D points.
        stm: ``(3, 3)`` homogeneous transformation matrix.

    Returns:
        ``(N, 2)`` transformed vertices (perspective-divided).
    """
    homogenous_vertices = torch.cat((vertices, torch.ones(vertices.shape[0], 1)), dim=1)
    transformed = torch.matmul(homogenous_vertices, stm)
    norm_factor = 1.0 / transformed[:, 2:]
    norm_factor[transformed[:, 2:] == 0] = 0
    return transformed[:, :2].contiguous() * norm_factor
