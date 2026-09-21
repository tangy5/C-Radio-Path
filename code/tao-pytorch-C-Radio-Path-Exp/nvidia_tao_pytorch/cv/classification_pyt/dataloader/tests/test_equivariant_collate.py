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

"""Unit tests for equivariant_collate."""

import pytest
import torch

from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.equivariant_collate import (
    equivariant_collate,
    NOCLASS_IDX,
)
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.defer_image import DeferImage
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.quad import Quad


def _make_sample(num_views, C=3, H=32, W=32, extras=None):
    """Create one sample: list of (DeferImage, Quad) pairs + optional extras."""
    torch.manual_seed(0)
    views = []
    for _ in range(num_views):
        img = torch.rand(C, H, W)
        views.append((DeferImage(img), Quad(img)))
    if extras is not None:
        views.extend(extras)
    return views


def _make_batch(batch_size, num_views, C=3, H=32, W=32, extras_per_sample=None):
    """Create a batch of samples."""
    batch = []
    for b in range(batch_size):
        torch.manual_seed(b)
        views = []
        for _ in range(num_views):
            img = torch.rand(C, H, W)
            views.append((DeferImage(img), Quad(img)))
        if extras_per_sample is not None:
            views.extend(extras_per_sample[b])
        batch.append(views)
    return batch


# =========================================================================
# Basic structure tests
# =========================================================================


@pytest.mark.cv_unit
def test_collate_student_only():
    """Single view (student only), no teachers."""
    collate_fn = equivariant_collate(num_images=1, patch_sizes=[14])
    batch = _make_batch(4, num_views=1)
    result = collate_fn(batch)

    assert "img" in result
    assert "class" in result
    assert "valid_mask" in result
    assert "teacher_views" in result

    assert result["img"].shape == (4, 3, 32, 32)
    assert result["valid_mask"].shape == (4, 32, 32)
    assert len(result["teacher_views"]) == 0
    assert torch.all(result["class"] == NOCLASS_IDX)


@pytest.mark.cv_unit
def test_collate_one_teacher():
    """Two views: student + one teacher."""
    collate_fn = equivariant_collate(num_images=2, patch_sizes=[14, 14])
    batch = _make_batch(4, num_views=2)
    result = collate_fn(batch)

    assert result["img"].shape == (4, 3, 32, 32)
    assert len(result["teacher_views"]) == 1

    tv = result["teacher_views"][0]
    assert tv["img"].shape == (4, 3, 32, 32)
    assert tv["valid_mask"].shape == (4, 32, 32)
    assert tv["spatial_transform"].shape == (4, 3, 3)


@pytest.mark.cv_unit
def test_collate_two_teachers():
    """Three views: student + two teachers."""
    collate_fn = equivariant_collate(num_images=3, patch_sizes=[14, 14, 16])
    batch = _make_batch(2, num_views=3)
    result = collate_fn(batch)

    assert result["img"].shape == (2, 3, 32, 32)
    assert len(result["teacher_views"]) == 2
    for tv in result["teacher_views"]:
        assert tv["spatial_transform"].shape == (2, 3, 3)


# =========================================================================
# Valid mask tests
# =========================================================================


@pytest.mark.cv_unit
def test_valid_mask_identity():
    """With identity transforms, valid mask should be all ones."""
    collate_fn = equivariant_collate(num_images=1, patch_sizes=[14])
    batch = _make_batch(2, num_views=1)
    result = collate_fn(batch)

    for b in range(2):
        assert result["valid_mask"][b].sum() == 32 * 32


@pytest.mark.cv_unit
def test_valid_mask_after_crop():
    """After a crop, valid mask should have fewer valid pixels."""
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.spatial import CropTransform
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.base import Size

    collate_fn = equivariant_collate(num_images=1, patch_sizes=[14])

    torch.manual_seed(0)
    img = torch.rand(3, 32, 32)
    di = DeferImage(img)
    q = Quad(img)
    CropTransform(4, 4, Size(24, 24))(di, q)

    batch = [[(di, q)]]
    result = collate_fn(batch)

    assert result["valid_mask"].shape == (1, 24, 24)
    valid_count = result["valid_mask"][0].sum().item()
    assert valid_count > 0
    assert valid_count <= 24 * 24


# =========================================================================
# Teacher-to-student transform tests
# =========================================================================


@pytest.mark.cv_unit
def test_teacher_transform_identity():
    """With identity transforms on both views, the teacher→student
    transform should be close to identity (in [-1,1] normalized coords)."""
    collate_fn = equivariant_collate(num_images=2, patch_sizes=[14, 14])
    batch = _make_batch(2, num_views=2)
    result = collate_fn(batch)

    stm = result["teacher_views"][0]["spatial_transform"]
    for b in range(2):
        assert torch.allclose(stm[b], torch.eye(3), atol=1e-5)


@pytest.mark.cv_unit
def test_teacher_transform_shape():
    """Transform matrix should be (B, 3, 3) float32."""
    collate_fn = equivariant_collate(num_images=2, patch_sizes=[14, 14])
    batch = _make_batch(3, num_views=2)
    result = collate_fn(batch)

    stm = result["teacher_views"][0]["spatial_transform"]
    assert stm.shape == (3, 3, 3)
    assert stm.dtype == torch.float32


# =========================================================================
# Extra fields
# =========================================================================


@pytest.mark.cv_unit
def test_extras_label():
    """Label extra should be placed in batch['class']."""
    extra_map = {0: "label"}
    collate_fn = equivariant_collate(
        num_images=1, patch_sizes=[14], extra_map=extra_map,
    )

    extras = [[5], [3], [7]]
    batch = _make_batch(3, num_views=1, extras_per_sample=extras)
    result = collate_fn(batch)

    assert torch.is_tensor(result["class"])
    assert result["class"].tolist() == [5, 3, 7]


@pytest.mark.cv_unit
def test_extras_multiple():
    """Multiple extras: label + image_key."""
    extra_map = {0: "image_key", 1: "label"}
    collate_fn = equivariant_collate(
        num_images=1, patch_sizes=[14], extra_map=extra_map,
    )

    extras = [["hash_a", 1], ["hash_b", 2]]
    batch = _make_batch(2, num_views=1, extras_per_sample=extras)
    result = collate_fn(batch)

    assert result["class"].tolist() == [1, 2]
    assert result["image_key"] == ["hash_a", "hash_b"]


@pytest.mark.cv_unit
def test_no_extras_gives_noclass():
    """Without extras, batch['class'] should be NOCLASS_IDX."""
    collate_fn = equivariant_collate(num_images=1, patch_sizes=[14])
    batch = _make_batch(2, num_views=1)
    result = collate_fn(batch)

    assert torch.all(result["class"] == NOCLASS_IDX)


# =========================================================================
# Integration: transforms + collation
# =========================================================================


@pytest.mark.cv_unit
def test_collate_with_transformed_views():
    """Apply different transforms to student and teacher, then collate."""
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.spatial import CropTransform
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.base import Size

    collate_fn = equivariant_collate(num_images=2, patch_sizes=[14, 14])

    torch.manual_seed(42)
    img = torch.rand(3, 64, 64)

    di_s = DeferImage(img.clone())
    q_s = Quad(img.clone())
    CropTransform(0, 0, Size(32, 32))(di_s, q_s)

    di_t = DeferImage(img.clone())
    q_t = Quad(img.clone())
    CropTransform(16, 16, Size(32, 32))(di_t, q_t)

    batch = [[(di_s, q_s), (di_t, q_t)]]
    result = collate_fn(batch)

    assert result["img"].shape == (1, 3, 32, 32)
    assert result["teacher_views"][0]["img"].shape == (1, 3, 32, 32)

    stm = result["teacher_views"][0]["spatial_transform"][0]
    assert not torch.allclose(stm, torch.eye(3), atol=1e-3)


@pytest.mark.cv_unit
def test_collate_batch_size_one():
    """Edge case: batch size of 1."""
    collate_fn = equivariant_collate(num_images=2, patch_sizes=[14, 14])
    batch = _make_batch(1, num_views=2)
    result = collate_fn(batch)

    assert result["img"].shape[0] == 1
    assert result["teacher_views"][0]["spatial_transform"].shape[0] == 1
