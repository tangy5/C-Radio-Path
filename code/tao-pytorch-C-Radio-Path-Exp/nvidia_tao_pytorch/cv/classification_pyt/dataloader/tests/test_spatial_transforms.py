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

"""Unit tests for Phase 4: base utilities, spatial transforms, and pipeline builder."""

import math

import numpy
import pytest
import torch

from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.base import (
    Size,
    RandomSize,
    ScopedRNG,
    Identity,
    RandomTransformBase,
    CompositeTransform,
    linear_sample,
    log_linear_sample,
    clamped_normal,
)
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.spatial import (
    CropTransform,
    RandomCropTransform,
    CenterCropTransform,
    ScaleTransform,
    MinSizeTransform,
    MaxSizeTransform,
    RandomZoomTransform,
    RandomFlipTransform,
    RandomTranslationTransform,
    RandomRotationTransform,
    PadToTransform,
)
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.pipeline import (
    get_pipeline,
    _get_zoom,
    _get_deformations,
)
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.defer_image import DeferImage
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.quad import Quad


# =========================================================================
# Base utilities
# =========================================================================


@pytest.mark.cv_unit
def test_size_basic():
    s = Size(224, 256)
    assert s.height == 224
    assert s.width == 256
    assert s() is s


@pytest.mark.cv_unit
def test_size_iter():
    s = Size(100, 200)
    h, w = s
    assert h == 100 and w == 200


@pytest.mark.cv_unit
def test_identity():
    image = torch.rand(3, 16, 16)
    di = DeferImage(image)
    stm_before = di.stm.clone()
    Identity()(di)
    assert torch.equal(di.stm, stm_before)


@pytest.mark.cv_unit
def test_linear_sample_range():
    rng = numpy.random.default_rng(42)
    for _ in range(50):
        v = linear_sample(2.0, 5.0, rng)
        assert 2.0 <= v <= 5.0


@pytest.mark.cv_unit
def test_log_linear_sample_range():
    rng = numpy.random.default_rng(42)
    for _ in range(50):
        v = log_linear_sample(0.5, 2.0, rng)
        assert 0.5 <= v <= 2.0


@pytest.mark.cv_unit
def test_clamped_normal_range():
    rng = numpy.random.default_rng(42)
    for _ in range(50):
        v = clamped_normal(1.0, -2.0, 2.0, rng)
        assert -2.0 <= v <= 2.0


@pytest.mark.cv_unit
def test_clamped_normal_equal_bounds():
    rng = numpy.random.default_rng(42)
    v = clamped_normal(1.0, 0.0, 0.0, rng)
    assert isinstance(v, float)


@pytest.mark.cv_unit
def test_scoped_rng_seed_determinism():
    a = ScopedRNG(seed=42)
    seed_a = a.seed
    b = ScopedRNG(seed=42)
    seed_b = b.seed
    assert seed_a == seed_b


@pytest.mark.cv_unit
def test_scoped_rng_reset():
    rng = ScopedRNG(seed=42)
    s1 = rng.seed
    rng.reset_seed()
    s2 = rng.seed
    assert s1 != s2


@pytest.mark.cv_unit
def test_random_size_deterministic():
    rng1 = ScopedRNG(seed=99)
    rs1 = RandomSize(rng1, base_size=224, min_size=192, max_size=256,
                     step_size=14, fixed_aspect=True)
    rng2 = ScopedRNG(seed=99)
    rs2 = RandomSize(rng2, base_size=224, min_size=192, max_size=256,
                     step_size=14, fixed_aspect=True)
    assert rs1() == rs2()


@pytest.mark.cv_unit
def test_random_size_explicit_resolutions():
    rng = ScopedRNG(seed=42)
    rs = RandomSize(rng, base_size=224, min_size=None, max_size=None,
                    step_size=14, resolutions={196: 0.5, 224: 0.5})
    s = rs()
    assert s.height in (196, 224)
    assert s.width == s.height


@pytest.mark.cv_unit
def test_composite_transform_applies_all():
    image = torch.rand(3, 64, 64)
    di = DeferImage(image)
    q = Quad(image)

    t1 = ScaleTransform(Size(128, 128))
    t2 = CenterCropTransform(Size(64, 64))

    ct = CompositeTransform([t1, t2])
    ct([di, q])

    assert not di.fresh


# =========================================================================
# Spatial transforms — each tested on (DeferImage, Quad)
# =========================================================================


def _make_items(C=3, H=64, W=96):
    image = torch.rand(C, H, W)
    return DeferImage(image), Quad(image)


@pytest.mark.cv_unit
def test_crop_transform():
    di, q = _make_items()
    ct = CropTransform(10, 5, Size(40, 50))
    ct(di, q)
    assert di.shape[-2:] == (40, 50)


@pytest.mark.cv_unit
def test_crop_transform_negative_raises():
    with pytest.raises(ValueError):
        CropTransform(-1, 0, Size(10, 10))


@pytest.mark.cv_unit
def test_random_crop_deterministic():
    image = torch.rand(3, 64, 96)
    di1 = DeferImage(image.clone())
    di2 = DeferImage(image.clone())
    q1 = Quad(image.clone())
    q2 = Quad(image.clone())

    rc1 = RandomCropTransform(Size(32, 48), random_seed=42)
    rc2 = RandomCropTransform(Size(32, 48), random_seed=42)

    rc1(di1, q1)
    rc2(di2, q2)

    assert torch.allclose(di1.stm, di2.stm)


@pytest.mark.cv_unit
def test_random_crop_with_quant():
    di, q = _make_items()
    rc = RandomCropTransform(Size(32, 48), random_seed=7, quant_shift=14)
    rc(di, q)
    assert not di.fresh


@pytest.mark.cv_unit
def test_center_crop():
    di, q = _make_items(H=100, W=200)
    cc = CenterCropTransform(Size(50, 80))
    cc(di, q)
    assert di.shape[-2:] == (50, 80)


@pytest.mark.cv_unit
def test_scale_transform():
    di, q = _make_items(H=50, W=100)
    st = ScaleTransform(Size(100, 200))
    st(di, q)
    assert abs(di.shape[-2] - 100.0) < 1e-4
    assert abs(di.shape[-1] - 200.0) < 1e-4


@pytest.mark.cv_unit
def test_min_size_noop_when_large():
    di, q = _make_items(H=200, W=300)
    ms = MinSizeTransform(100, 100)
    stm_before = di.stm.clone()
    ms(di, q)
    assert torch.equal(di.stm, stm_before)


@pytest.mark.cv_unit
def test_min_size_scales_up():
    di, q = _make_items(H=50, W=80)
    ms = MinSizeTransform(160, 100)
    ms(di, q)
    assert not di.fresh


@pytest.mark.cv_unit
def test_max_size_largest():
    di, q = _make_items(H=100, W=200)
    mx = MaxSizeTransform(Size(50, 50), smallest=False)
    mx(di, q)
    assert not di.fresh


@pytest.mark.cv_unit
def test_max_size_smallest():
    di, q = _make_items(H=100, W=200)
    mx = MaxSizeTransform(Size(50, 50), smallest=True)
    mx(di, q)
    assert not di.fresh


@pytest.mark.cv_unit
def test_random_zoom_deterministic():
    image = torch.rand(3, 64, 64)
    di1 = DeferImage(image.clone())
    di2 = DeferImage(image.clone())

    rz1 = RandomZoomTransform(0.8, 1.2, random_seed=42)
    rz2 = RandomZoomTransform(0.8, 1.2, random_seed=42)

    rz1(di1)
    rz2(di2)

    assert torch.allclose(di1.stm, di2.stm)


@pytest.mark.cv_unit
def test_random_zoom_log_scale():
    """Log-linear sampling when range spans 1.0."""
    di, _ = _make_items(H=64, W=64)
    rz = RandomZoomTransform(0.5, 2.0, random_seed=42)
    rz(di)
    assert not di.fresh


@pytest.mark.cv_unit
def test_random_flip_deterministic():
    image = torch.rand(3, 64, 64)
    di1 = DeferImage(image.clone())
    di2 = DeferImage(image.clone())

    rf1 = RandomFlipTransform(prob_apply=0.5, random_seed=42)
    rf2 = RandomFlipTransform(prob_apply=0.5, random_seed=42)

    rf1(di1)
    rf2(di2)

    assert torch.allclose(di1.stm, di2.stm)


@pytest.mark.cv_unit
def test_random_flip_always():
    di, _ = _make_items(H=64, W=64)
    rf = RandomFlipTransform(prob_apply=1.0, random_seed=42)
    rf(di)
    assert not di.fresh


@pytest.mark.cv_unit
def test_random_flip_never():
    di, _ = _make_items(H=64, W=64)
    rf = RandomFlipTransform(prob_apply=0.0, random_seed=42)
    stm_before = di.stm.clone()
    rf(di)
    assert torch.equal(di.stm, stm_before)


@pytest.mark.cv_unit
def test_random_translation():
    di, _ = _make_items(H=64, W=64)
    rt = RandomTranslationTransform(std_dev=5.0, abs_max=10, random_seed=42)
    rt(di)
    assert not di.fresh


@pytest.mark.cv_unit
def test_random_rotation():
    di, q = _make_items(H=64, W=64)
    rr = RandomRotationTransform(abs_max=30, random_seed=42)
    rr(di, q)
    assert not di.fresh


@pytest.mark.cv_unit
def test_pad_to_noop():
    di, q = _make_items(H=64, W=96)
    pt = PadToTransform(Size(32, 48), random_seed=42)
    stm_before = di.stm.clone()
    pt(di, q)
    assert torch.equal(di.stm, stm_before)


@pytest.mark.cv_unit
def test_pad_to_enlarges():
    di, q = _make_items(H=32, W=32)
    pt = PadToTransform(Size(64, 64), random_seed=42, rand_pad=False)
    pt(di, q)
    assert di.shape[-2:] == (64, 64)


@pytest.mark.cv_unit
def test_pad_to_with_quant():
    di, q = _make_items(H=32, W=32)
    pt = PadToTransform(Size(64, 64), random_seed=42, quant_pad=14, rand_pad=True)
    pt(di, q)
    assert di.shape[-2:] == (64, 64)


# =========================================================================
# Pipeline builder
# =========================================================================


@pytest.mark.cv_unit
def test_get_zoom_identity():
    t = _get_zoom(1.0)
    assert isinstance(t, Identity)


@pytest.mark.cv_unit
def test_get_zoom_float():
    t = _get_zoom(0.8, random_seed=1)
    assert isinstance(t, RandomZoomTransform)


@pytest.mark.cv_unit
def test_get_zoom_tuple():
    t = _get_zoom((0.8, 1.2), random_seed=1)
    assert isinstance(t, RandomZoomTransform)


@pytest.mark.cv_unit
def test_get_pipeline_train_student():
    rng = numpy.random.default_rng(42)
    pipeline = get_pipeline(
        student_size=224, img_size=224, patch_size=14,
        is_train=True, is_teacher=False,
        max_img_size=256, rng=rng, unified_seed=100,
    )
    assert isinstance(pipeline, CompositeTransform)
    assert len(pipeline.transforms) > 0


@pytest.mark.cv_unit
def test_get_pipeline_teacher():
    rng = numpy.random.default_rng(42)
    pipeline = get_pipeline(
        student_size=224, img_size=256, patch_size=14,
        is_train=False, is_teacher=True,
        max_img_size=256, rng=rng, unified_seed=100,
    )
    assert isinstance(pipeline, CompositeTransform)


@pytest.mark.cv_unit
def test_get_pipeline_hi_res():
    rng = numpy.random.default_rng(42)
    pipeline = get_pipeline(
        student_size=224, img_size=224, patch_size=14,
        is_train=True, is_teacher=False,
        max_img_size=1024, rng=rng, unified_seed=100,
    )
    assert isinstance(pipeline, CompositeTransform)


@pytest.mark.cv_unit
def test_get_pipeline_perf_test():
    rng = numpy.random.default_rng(42)
    pipeline = get_pipeline(
        student_size=224, img_size=224, patch_size=14,
        is_train=True, is_teacher=False,
        max_img_size=256, rng=rng, unified_seed=100,
        perf_test_simple_aug=True,
    )
    assert len(pipeline.transforms) == 2


@pytest.mark.cv_unit
def test_get_pipeline_full_equivariance():
    rng = numpy.random.default_rng(42)
    pipeline = get_pipeline(
        student_size=224, img_size=224, patch_size=14,
        is_train=True, is_teacher=False,
        max_img_size=256, rng=rng, unified_seed=100,
        full_equivariance=True,
    )
    assert isinstance(pipeline, CompositeTransform)


@pytest.mark.cv_unit
def test_get_pipeline_shift_equivariance_teacher():
    rng = numpy.random.default_rng(42)
    pipeline = get_pipeline(
        student_size=224, img_size=256, patch_size=14,
        is_train=False, is_teacher=True,
        max_img_size=256, rng=rng, unified_seed=100,
        shift_equivariance=True,
    )
    assert isinstance(pipeline, CompositeTransform)


@pytest.mark.cv_unit
def test_get_pipeline_applies_to_defer_image():
    """End-to-end: build pipeline, apply to (DeferImage, Quad)."""
    rng = numpy.random.default_rng(42)
    pipeline = get_pipeline(
        student_size=224, img_size=224, patch_size=14,
        is_train=True, is_teacher=False,
        max_img_size=256, rng=rng, unified_seed=100,
    )
    image = torch.rand(3, 300, 400)
    di = DeferImage(image)
    q = Quad(image)

    pipeline([di, q])

    assert not di.fresh
    assert di.stm.shape == (3, 3)
