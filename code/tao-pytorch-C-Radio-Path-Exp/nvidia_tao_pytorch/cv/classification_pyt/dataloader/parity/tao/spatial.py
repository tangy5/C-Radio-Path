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

"""Reference parity dump for spatial transforms and pipeline builder."""

import numpy

import torch

from nvidia_tao_pytorch.cv.classification_pyt.dataloader.parity import component


def _make_image(C=3, H=120, W=160):
    torch.manual_seed(7)
    return torch.rand(C, H, W)


def _snapshot_di(di):
    return {
        "stm": di.stm.flatten().tolist(),
        "target_size": list(di.target_size),
        "fresh": di.fresh,
    }


def _snapshot_q(q):
    return q.bounds.flatten().tolist()


# -----------------------------------------------------------------------
# Individual transforms
# -----------------------------------------------------------------------


@component("scale_transform")
def dump_scale_transform():
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.base import Size
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.spatial import ScaleTransform
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.defer_image import DeferImage
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.quad import Quad

    image = _make_image()
    di = DeferImage(image)
    q = Quad(image)

    st = ScaleTransform(Size(60, 80))
    st(di, q)

    return {"di": _snapshot_di(di), "q": _snapshot_q(q)}


@component("max_size_transform")
def dump_max_size_transform():
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.base import Size
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.spatial import MaxSizeTransform
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.defer_image import DeferImage
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.quad import Quad

    results = {}
    image = _make_image()

    di1 = DeferImage(image.clone())
    q1 = Quad(image.clone())
    MaxSizeTransform(Size(80, 80), smallest=False)(di1, q1)
    results["largest"] = {"di": _snapshot_di(di1), "q": _snapshot_q(q1)}

    di2 = DeferImage(image.clone())
    q2 = Quad(image.clone())
    MaxSizeTransform(Size(80, 80), smallest=True)(di2, q2)
    results["smallest"] = {"di": _snapshot_di(di2), "q": _snapshot_q(q2)}

    return results


@component("center_crop_transform")
def dump_center_crop_transform():
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.base import Size
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.spatial import CenterCropTransform
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.defer_image import DeferImage
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.quad import Quad

    image = _make_image()
    di = DeferImage(image)
    q = Quad(image)

    CenterCropTransform(Size(60, 80))(di, q)

    return {"di": _snapshot_di(di), "q": _snapshot_q(q)}


@component("random_crop_transform")
def dump_random_crop_transform():
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.base import Size
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.spatial import RandomCropTransform
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.defer_image import DeferImage
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.quad import Quad

    results = {}
    image = _make_image()

    di1 = DeferImage(image.clone())
    q1 = Quad(image.clone())
    RandomCropTransform(Size(60, 80), random_seed=42)(di1, q1)
    results["basic"] = {"di": _snapshot_di(di1), "q": _snapshot_q(q1)}

    di2 = DeferImage(image.clone())
    q2 = Quad(image.clone())
    RandomCropTransform(Size(60, 80), random_seed=42, quant_shift=14)(di2, q2)
    results["quant"] = {"di": _snapshot_di(di2), "q": _snapshot_q(q2)}

    return results


@component("random_zoom_transform")
def dump_random_zoom_transform():
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.spatial import RandomZoomTransform
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.defer_image import DeferImage
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.quad import Quad

    results = {}
    image = _make_image(H=64, W=64)

    di1 = DeferImage(image.clone())
    q1 = Quad(image.clone())
    RandomZoomTransform(0.8, 1.2, random_seed=42)(di1, q1)
    results["linear"] = {"di": _snapshot_di(di1), "q": _snapshot_q(q1)}

    di2 = DeferImage(image.clone())
    q2 = Quad(image.clone())
    RandomZoomTransform(0.5, 2.0, random_seed=42)(di2, q2)
    results["log_linear"] = {"di": _snapshot_di(di2), "q": _snapshot_q(q2)}

    return results


@component("random_flip_transform")
def dump_random_flip_transform():
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.spatial import RandomFlipTransform
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.defer_image import DeferImage
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.quad import Quad

    image = _make_image(H=64, W=64)
    di = DeferImage(image)
    q = Quad(image)

    RandomFlipTransform(prob_apply=1.0, random_seed=42)(di, q)

    return {"di": _snapshot_di(di), "q": _snapshot_q(q)}


@component("random_rotation_transform")
def dump_random_rotation_transform():
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.spatial import RandomRotationTransform
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.defer_image import DeferImage
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.quad import Quad

    image = _make_image(H=64, W=64)
    di = DeferImage(image)
    q = Quad(image)

    RandomRotationTransform(abs_max=30, random_seed=42)(di, q)

    return {"di": _snapshot_di(di), "q": _snapshot_q(q)}


@component("pad_to_transform")
def dump_pad_to_transform():
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.base import Size
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.spatial import PadToTransform
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.defer_image import DeferImage
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.quad import Quad

    results = {}
    image = _make_image(H=32, W=32)

    di1 = DeferImage(image.clone())
    q1 = Quad(image.clone())
    PadToTransform(Size(64, 64), random_seed=42, rand_pad=False)(di1, q1)
    results["fixed"] = {"di": _snapshot_di(di1), "q": _snapshot_q(q1)}

    di2 = DeferImage(image.clone())
    q2 = Quad(image.clone())
    PadToTransform(Size(64, 64), random_seed=42, rand_pad=True)(di2, q2)
    results["random"] = {"di": _snapshot_di(di2), "q": _snapshot_q(q2)}

    return results


# -----------------------------------------------------------------------
# Full pipeline
# -----------------------------------------------------------------------


@component("get_pipeline")
def dump_get_pipeline():
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.pipeline import get_pipeline
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.defer_image import DeferImage
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.quad import Quad

    results = {}

    image = _make_image(H=300, W=400)

    rng = numpy.random.default_rng(42)
    pipeline = get_pipeline(
        student_size=224, img_size=224, patch_size=14,
        is_train=True, is_teacher=False,
        max_img_size=256, rng=rng, unified_seed=100,
    )
    di = DeferImage(image.clone())
    q = Quad(image.clone())
    pipeline([di, q])
    results["train_student_lores"] = {"di": _snapshot_di(di), "q": _snapshot_q(q)}

    rng = numpy.random.default_rng(42)
    pipeline = get_pipeline(
        student_size=224, img_size=256, patch_size=14,
        is_train=False, is_teacher=True,
        max_img_size=256, rng=rng, unified_seed=100,
    )
    di = DeferImage(image.clone())
    q = Quad(image.clone())
    pipeline([di, q])
    results["teacher_lores"] = {"di": _snapshot_di(di), "q": _snapshot_q(q)}

    rng = numpy.random.default_rng(42)
    pipeline = get_pipeline(
        student_size=224, img_size=224, patch_size=14,
        is_train=True, is_teacher=False,
        max_img_size=1024, rng=rng, unified_seed=100,
    )
    di = DeferImage(image.clone())
    q = Quad(image.clone())
    pipeline([di, q])
    results["train_student_hires"] = {"di": _snapshot_di(di), "q": _snapshot_q(q)}

    rng = numpy.random.default_rng(42)
    pipeline = get_pipeline(
        student_size=224, img_size=224, patch_size=14,
        is_train=True, is_teacher=False,
        max_img_size=256, rng=rng, unified_seed=100,
        full_equivariance=True,
    )
    di = DeferImage(image.clone())
    q = Quad(image.clone())
    pipeline([di, q])
    results["train_full_equivariance"] = {"di": _snapshot_di(di), "q": _snapshot_q(q)}

    rng = numpy.random.default_rng(42)
    pipeline = get_pipeline(
        student_size=224, img_size=256, patch_size=14,
        is_train=False, is_teacher=True,
        max_img_size=256, rng=rng, unified_seed=100,
        shift_equivariance=True,
    )
    di = DeferImage(image.clone())
    q = Quad(image.clone())
    pipeline([di, q])
    results["teacher_shift_equivariance"] = {"di": _snapshot_di(di), "q": _snapshot_q(q)}

    return results
