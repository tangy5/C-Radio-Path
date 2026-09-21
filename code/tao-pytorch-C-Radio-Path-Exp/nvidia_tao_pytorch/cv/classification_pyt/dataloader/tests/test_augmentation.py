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

"""Unit tests for CLDataAugmentation transforms."""

import random

import numpy as np
import pytest
import torch
from PIL import Image

from nvidia_tao_pytorch.cv.classification_pyt.dataloader.augmentation import (
    CLDataAugmentation,
)


def _make_pil_image(width, height, seed=0):
    """Create a deterministic RGB PIL image with the given dimensions."""
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _make_val_augmentor(img_size=224):
    return CLDataAugmentation(img_size=img_size, is_training=False)


def _make_train_augmentor(img_size=224):
    return CLDataAugmentation(img_size=img_size, is_training=True)


# ---------------------------------------------------------------------------
# Val transform tests
# ---------------------------------------------------------------------------


@pytest.mark.cv_unit
def test_val_aspect_resize_landscape():
    aug = _make_val_augmentor(img_size=224)
    img = _make_pil_image(640, 480)
    [out] = aug.transform([img])
    assert out.shape == (3, 224, 224)
    assert out.min() >= 0.0 and out.max() <= 1.0


@pytest.mark.cv_unit
def test_val_aspect_resize_portrait():
    aug = _make_val_augmentor(img_size=224)
    img = _make_pil_image(480, 640)
    [out] = aug.transform([img])
    assert out.shape == (3, 224, 224)
    assert out.min() >= 0.0 and out.max() <= 1.0


@pytest.mark.cv_unit
def test_val_aspect_resize_square():
    aug = _make_val_augmentor(img_size=224)
    img = _make_pil_image(256, 256)
    [out] = aug.transform([img])
    assert out.shape == (3, 224, 224)


@pytest.mark.cv_unit
def test_val_aspect_resize_extreme_ratio():
    aug = _make_val_augmentor(img_size=224)
    img = _make_pil_image(1000, 100)
    [out] = aug.transform([img])
    assert out.shape == (3, 224, 224)


@pytest.mark.cv_unit
def test_val_deterministic():
    """Val transforms should be deterministic (no stochastic augmentations)."""
    aug = _make_val_augmentor(img_size=224)
    img = _make_pil_image(640, 480, seed=42)
    [out1] = aug.transform([img.copy()])
    [out2] = aug.transform([img.copy()])
    assert torch.equal(out1, out2)


@pytest.mark.cv_unit
def test_val_output_range_zero_one():
    aug = _make_val_augmentor(img_size=224)
    img = _make_pil_image(300, 400, seed=7)
    [out] = aug.transform([img])
    assert out.dtype == torch.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0


# ---------------------------------------------------------------------------
# Train transform tests
# ---------------------------------------------------------------------------


@pytest.mark.cv_unit
def test_train_aspect_resize_and_random_crop():
    aug = _make_train_augmentor(img_size=224)
    img = _make_pil_image(640, 480)
    [out] = aug.transform([img])
    assert out.shape == (3, 224, 224)
    assert out.min() >= 0.0 and out.max() <= 1.0


@pytest.mark.cv_unit
def test_train_vs_val_different_outputs():
    """Train (random crop) and val (center crop) should differ on non-square images."""
    img_size = 224
    train_aug = _make_train_augmentor(img_size=img_size)
    val_aug = _make_val_augmentor(img_size=img_size)
    img = _make_pil_image(640, 480, seed=99)

    val_out_list = []
    for _ in range(5):
        [v] = val_aug.transform([img.copy()])
        val_out_list.append(v)
    # All val outputs should be identical (deterministic)
    for v in val_out_list[1:]:
        assert torch.equal(val_out_list[0], v)

    # At least one train output should differ from val (random crop)
    random.seed(None)
    found_diff = False
    for _ in range(10):
        [t] = train_aug.transform([img.copy()])
        if not torch.equal(t, val_out_list[0]):
            found_diff = True
            break
    assert found_diff, "Expected train and val to produce different crops"


@pytest.mark.cv_unit
def test_train_square_input_passthrough():
    """If input is already img_size x img_size, resize+crop is a no-op."""
    aug = _make_train_augmentor(img_size=224)
    img = _make_pil_image(224, 224, seed=5)
    [out] = aug.transform([img])
    assert out.shape == (3, 224, 224)


# ---------------------------------------------------------------------------
# Regression / error tests
# ---------------------------------------------------------------------------


@pytest.mark.cv_unit
def test_img_size_required():
    with pytest.raises(ValueError, match="img_size must be specified"):
        CLDataAugmentation(img_size=None)
