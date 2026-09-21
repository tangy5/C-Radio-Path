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

"""Unit tests for Quad, generate_homography_grid, and DeferImage."""

import math

import pytest
import torch

from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.quad import (
    Quad,
    _apply_single_stm,
)
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.generate_homography_grid import (
    generate_homography_grid,
    BASE_GRID_CACHE,
)
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.defer_image import (
    DeferImage,
)
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.ops.spatial_transform import (
    spatial_transform,
    spatial_transform_fallback,
    _cpp_spatial_transform,
)


# ---------------------------------------------------------------------------
# _apply_single_stm
# ---------------------------------------------------------------------------


@pytest.mark.cv_unit
def test_apply_stm_identity():
    verts = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    stm = torch.eye(3)
    result = _apply_single_stm(verts, stm)
    assert torch.allclose(result, verts)


@pytest.mark.cv_unit
def test_apply_stm_translation():
    verts = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    stm = torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
        [5, 10, 1],
    ], dtype=torch.float32)
    result = _apply_single_stm(verts, stm)
    expected = torch.tensor([[5.0, 10.0], [6.0, 11.0]])
    assert torch.allclose(result, expected)


@pytest.mark.cv_unit
def test_apply_stm_scale():
    verts = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    stm = torch.tensor([
        [2, 0, 0],
        [0, 3, 0],
        [0, 0, 1],
    ], dtype=torch.float32)
    result = _apply_single_stm(verts, stm)
    expected = torch.tensor([[2.0, 6.0], [6.0, 12.0]])
    assert torch.allclose(result, expected)


@pytest.mark.cv_unit
def test_apply_stm_zero_w():
    """When w=0 in homogeneous coords, norm_factor should be 0."""
    verts = torch.tensor([[1.0, 0.0]])
    stm = torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ], dtype=torch.float32)
    result = _apply_single_stm(verts, stm)
    assert torch.allclose(result, torch.zeros(1, 2))


# ---------------------------------------------------------------------------
# Quad
# ---------------------------------------------------------------------------


@pytest.mark.cv_unit
def test_quad_init():
    image = torch.zeros(3, 100, 200)  # C, H, W
    q = Quad(image)
    expected = torch.tensor([
        [0, 0], [200, 0], [200, 100], [0, 100]
    ], dtype=torch.float32)
    assert torch.allclose(q.bounds, expected)


@pytest.mark.cv_unit
def test_quad_translate():
    image = torch.zeros(3, 50, 80)
    q = Quad(image)
    q.translate(torch.tensor([10.0, 20.0]))
    assert q.bounds[0, 0].item() == 10.0
    assert q.bounds[0, 1].item() == 20.0


@pytest.mark.cv_unit
def test_quad_scale():
    image = torch.zeros(3, 50, 80)
    q = Quad(image)
    q.scale(torch.tensor([2.0, 3.0]))
    assert q.bounds[1, 0].item() == 160.0  # 80 * 2
    assert q.bounds[2, 1].item() == 150.0  # 50 * 3


@pytest.mark.cv_unit
def test_quad_flip():
    image = torch.zeros(3, 50, 80)
    q = Quad(image)
    q.flip(40.0)
    assert q.bounds[0, 0].item() == 80.0
    assert q.bounds[1, 0].item() == 0.0


@pytest.mark.cv_unit
def test_quad_rotate():
    image = torch.zeros(3, 100, 100)
    q = Quad(image)
    angle = math.pi / 4
    rot = torch.tensor([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ], dtype=torch.float32)
    q.rotate(rot)
    assert q.bounds.shape == (4, 2)


@pytest.mark.cv_unit
def test_quad_apply_stm():
    image = torch.zeros(3, 100, 100)
    q = Quad(image)
    stm = torch.eye(3)
    stm[0, 0] = 2.0
    q.apply_stm(stm)
    assert q.bounds[1, 0].item() == 200.0


# ---------------------------------------------------------------------------
# generate_homography_grid
# ---------------------------------------------------------------------------


@pytest.mark.cv_unit
def test_homography_grid_identity():
    BASE_GRID_CACHE.clear()
    N, C, H, W = 1, 3, 8, 8
    hom = torch.eye(3).unsqueeze(0)
    grid = generate_homography_grid(hom, (N, C, H, W))
    assert grid.shape == (1, 8, 8, 2)
    assert grid[0, 0, 0, 0].item() == pytest.approx(-1.0, abs=1e-5)
    assert grid[0, 0, 0, 1].item() == pytest.approx(-1.0, abs=1e-5)
    assert grid[0, -1, -1, 0].item() == pytest.approx(1.0, abs=1e-5)
    assert grid[0, -1, -1, 1].item() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.cv_unit
def test_homography_grid_cache():
    BASE_GRID_CACHE.clear()
    size = (1, 3, 4, 4)
    hom = torch.eye(3).unsqueeze(0)
    generate_homography_grid(hom, size)
    assert size in BASE_GRID_CACHE
    generate_homography_grid(hom, size)
    assert size in BASE_GRID_CACHE


@pytest.mark.cv_unit
def test_homography_grid_batch():
    BASE_GRID_CACHE.clear()
    N = 4
    hom = torch.eye(3).unsqueeze(0).expand(N, -1, -1).contiguous()
    grid = generate_homography_grid(hom, (N, 3, 8, 8))
    assert grid.shape == (N, 8, 8, 2)
    for i in range(N):
        assert torch.allclose(grid[0], grid[i])


@pytest.mark.cv_unit
def test_homography_grid_scale():
    BASE_GRID_CACHE.clear()
    hom = torch.tensor([
        [0.5, 0, 0],
        [0, 0.5, 0],
        [0, 0, 1],
    ], dtype=torch.float32).unsqueeze(0)
    grid = generate_homography_grid(hom, (1, 3, 8, 8))
    assert grid[0, 0, 0, 0].item() == pytest.approx(-0.5, abs=1e-5)


# ---------------------------------------------------------------------------
# spatial_transform
# ---------------------------------------------------------------------------


@pytest.mark.cv_unit
def test_spatial_transform_identity():
    image = torch.rand(1, 3, 16, 16)
    stm = torch.eye(3).unsqueeze(0)
    result = spatial_transform(image, stm, 16, 16)
    assert result.shape == (1, 3, 16, 16)


@pytest.mark.cv_unit
def test_spatial_transform_resize():
    image = torch.rand(1, 3, 16, 16)
    stm = torch.eye(3).unsqueeze(0)
    result = spatial_transform(image, stm, 32, 32)
    assert result.shape == (1, 3, 32, 32)


@pytest.mark.cv_unit
def test_spatial_transform_pad_color():
    """Non-zero background should fill out-of-bounds regions."""
    image = torch.ones(1, 1, 4, 4)
    hom = torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
        [5, 5, 1],
    ], dtype=torch.float32).unsqueeze(0)
    result = spatial_transform(image, hom, 4, 4, background=0.5)
    assert result.shape == (1, 1, 4, 4)


@pytest.mark.cv_unit
def test_spatial_transform_cpp_matches_fallback():
    """If C++ extension is available, its output must match the Python fallback."""
    if _cpp_spatial_transform is None:
        pytest.skip("C++ SpatialTransformOps not built")

    torch.manual_seed(0)
    image = torch.rand(2, 3, 32, 32)

    stm_identity = torch.eye(3).unsqueeze(0).expand(2, -1, -1).contiguous()
    cpp_out = _cpp_spatial_transform(image, stm_identity, 32, 32, "bilinear", 0.0, False)
    py_out = spatial_transform_fallback(image, stm_identity, 32, 32)
    assert torch.allclose(cpp_out, py_out, rtol=1e-5, atol=1e-5)

    stm_translate = torch.eye(3).unsqueeze(0).expand(2, -1, -1).clone()
    stm_translate[:, 2, 0] = 3.0
    stm_translate[:, 2, 1] = -2.0
    cpp_out = _cpp_spatial_transform(image, stm_translate, 32, 32, "bilinear", 0.0, False)
    py_out = spatial_transform_fallback(image, stm_translate, 32, 32)
    assert torch.allclose(cpp_out, py_out, rtol=1e-5, atol=1e-5)

    stm_scale = torch.tensor([
        [0.5, 0, 0],
        [0, 0.5, 0],
        [0, 0, 1],
    ], dtype=torch.float32).unsqueeze(0).expand(2, -1, -1).contiguous()
    cpp_out = _cpp_spatial_transform(image, stm_scale, 16, 16, "bilinear", 0.0, False)
    py_out = spatial_transform_fallback(image, stm_scale, 16, 16)
    assert torch.allclose(cpp_out, py_out, rtol=1e-5, atol=1e-5)


@pytest.mark.cv_unit
def test_spatial_transform_cpp_pad_color_matches_fallback():
    """If C++ extension is available, non-zero background should match fallback."""
    if _cpp_spatial_transform is None:
        pytest.skip("C++ SpatialTransformOps not built")

    image = torch.ones(1, 1, 4, 4)
    stm = torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
        [5, 5, 1],
    ], dtype=torch.float32).unsqueeze(0)
    cpp_out = _cpp_spatial_transform(image, stm, 4, 4, "bilinear", 0.5, False)
    py_out = spatial_transform_fallback(image, stm, 4, 4, pad_color=0.5)
    assert torch.allclose(cpp_out, py_out, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# DeferImage
# ---------------------------------------------------------------------------


@pytest.mark.cv_unit
def test_defer_image_fresh():
    """A fresh DeferImage should return the original tensor unchanged."""
    image = torch.rand(3, 16, 16)
    di = DeferImage(image)
    result = di()
    assert torch.equal(result, image)


@pytest.mark.cv_unit
def test_defer_image_shape_property():
    image = torch.rand(3, 32, 64)
    di = DeferImage(image)
    assert di.shape == (3, 32, 64)
    assert di.dtype == image.dtype


@pytest.mark.cv_unit
def test_defer_image_scale():
    image = torch.rand(3, 16, 16)
    di = DeferImage(image)
    di.scale(torch.tensor([2.0, 2.0]))
    assert di.shape[-2:] == (32.0, 32.0)
    assert not di.fresh


@pytest.mark.cv_unit
def test_defer_image_translate():
    image = torch.rand(3, 16, 16)
    di = DeferImage(image)
    di.translate(torch.tensor([5.0, 5.0]))
    assert not di.fresh


@pytest.mark.cv_unit
def test_defer_image_flip():
    image = torch.rand(3, 16, 16)
    di = DeferImage(image)
    di.flip(8.0)
    assert not di.fresh


@pytest.mark.cv_unit
def test_defer_image_rotate():
    image = torch.rand(3, 16, 16)
    di = DeferImage(image)
    angle = math.pi / 6
    rot_mat = torch.tensor([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ], dtype=torch.float32)
    di.rotate(rot_mat)
    assert not di.fresh
    assert di.shape[-2] != 16 or di.shape[-1] != 16


@pytest.mark.cv_unit
def test_defer_image_clip_translate():
    image = torch.rand(3, 16, 16)
    di = DeferImage(image)
    di.clip_translate(2, 2, 8, 8)
    assert di.shape[-2:] == (8, 8)
    assert not di.fresh


@pytest.mark.cv_unit
def test_defer_image_apply_stm():
    image = torch.rand(3, 16, 16)
    di = DeferImage(image)
    stm = torch.eye(3)
    di.apply_stm(stm)
    assert not di.fresh


@pytest.mark.cv_unit
def test_defer_image_call_3d():
    """__call__ on a 3D image should return a 3D tensor."""
    image = torch.rand(3, 16, 16)
    di = DeferImage(image, allow_eager=False)
    result = di()
    assert result.dim() == 3
    assert result.shape == (3, 16, 16)


@pytest.mark.cv_unit
def test_defer_image_call_4d():
    """__call__ on a 4D image should return a 4D tensor."""
    image = torch.rand(2, 3, 16, 16)
    di = DeferImage(image, allow_eager=False)
    result = di()
    assert result.dim() == 4


@pytest.mark.cv_unit
def test_defer_image_uint8_conversion():
    """uint8 images should be converted to float and divided by 255."""
    image = torch.randint(0, 256, (3, 8, 8), dtype=torch.uint8)
    di = DeferImage(image, allow_eager=False)
    result = di()
    assert result.dtype == torch.float32
    assert result.max() <= 1.0


@pytest.mark.cv_unit
def test_defer_image_can_collate():
    a = DeferImage(torch.rand(3, 16, 16))
    b = DeferImage(torch.rand(3, 16, 16))
    assert a.can_collate(b)

    c = DeferImage(torch.rand(3, 32, 32))
    assert not a.can_collate(c)

    assert not a.can_collate("not a DeferImage")


@pytest.mark.cv_unit
def test_defer_image_collate():
    a = DeferImage(torch.rand(3, 8, 8))
    b = DeferImage(torch.rand(3, 8, 8))
    c = DeferImage(torch.rand(3, 8, 8))
    a.collate([b, c])
    assert a.image.shape == (3, 3, 8, 8)
    assert a.stm.shape == (3, 3, 3)


@pytest.mark.cv_unit
def test_defer_image_get_bds():
    image = torch.rand(3, 32, 64)
    di = DeferImage(image)
    bds = di.get_bds()
    assert bds.shape == (4, 2)


@pytest.mark.cv_unit
def test_defer_image_stm_accumulation():
    """Multiple transforms should accumulate via matrix multiplication."""
    image = torch.rand(3, 16, 16)
    di = DeferImage(image)
    stm1 = torch.eye(3)
    stm1[2, 0] = 3.0
    stm2 = torch.eye(3)
    stm2[0, 0] = 2.0
    di.apply_stm(stm1)
    di.apply_stm(stm2)
    expected = stm1 @ stm2
    assert torch.allclose(di.stm, expected)


@pytest.mark.cv_unit
def test_defer_image_scale_retain_dims():
    image = torch.rand(3, 16, 16)
    di = DeferImage(image)
    di.scale(torch.tensor([2.0, 2.0]), retain_dims=True)
    assert di.shape[-2:] == (16, 16)
