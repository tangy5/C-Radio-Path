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

"""Transform pipeline builder.

``get_pipeline()`` assembles a ``CompositeTransform`` chain for a single
view (student or teacher).  It handles:

  - Hi-res vs lo-res paths (threshold: 512 px)
  - Unified vs individual deformations
  - Stochastic resolutions with per-teacher patch-size rescaling
  - Shift equivariance (jittered teacher crops)
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from timm.layers import to_2tuple

from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.base import (
    CompositeTransform,
    Identity,
    RandomSize,
    ScopedRNG,
    Size,
)
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.spatial import (
    CenterCropTransform,
    MaxSizeTransform,
    PadToTransform,
    RandomCropTransform,
    RandomFlipTransform,
    RandomPerspectiveTransform,
    RandomRotationTransform,
    RandomZoomTransform,
)

logger = logging.getLogger(__name__)


def _get_zoom(scale: Union[float, Tuple[float, float]] = 1.0, random_seed: int = None):
    """Return a RandomZoomTransform or Identity depending on *scale*."""
    if isinstance(scale, float) and scale != 1.0:
        return RandomZoomTransform(min_ratio=scale, max_ratio=1 / scale, retain_dims=False, random_seed=random_seed)
    if isinstance(scale, tuple):
        return RandomZoomTransform(min_ratio=scale[0], max_ratio=scale[1], retain_dims=False, random_seed=random_seed)
    return Identity()


def _get_deformations(crop_size: Union[Size, RandomSize],
                      rng: np.random.Generator,
                      unified_seed: int,
                      unified_scale: Union[float, Tuple[float, float]] = 1.0,
                      unified_stretch: float = 1.0,
                      flip_prob: float = 0.0,
                      rot_max: float = 0.0,
                      perspective_scale: Tuple[float, float] = (0, 0),
                      individual_scale: Union[float, Tuple[float, float]] = 1.0,
                      patch_size: int = None,
                      jitter: bool = False,
                      full_equivariance: bool = False,
                      ):
    """Build the deformation sub-pipeline (zoom, crop, flip, rotate, perspective)."""
    ret = []

    jitter = jitter and not full_equivariance

    def get_seed():
        return rng.bit_generator.random_raw()

    ret.append(_get_zoom(unified_scale, random_seed=unified_seed + 41))

    if unified_stretch != 1.0:
        ret.extend([
            RandomZoomTransform(min_ratio=unified_stretch, max_ratio=1 / unified_stretch, fixed_width=True, retain_dims=False, random_seed=unified_seed + 42),
            RandomZoomTransform(min_ratio=unified_stretch, max_ratio=1 / unified_stretch, fixed_height=True, retain_dims=False, random_seed=unified_seed + 43),
        ])

    ret.append(RandomCropTransform(image_size=crop_size, random_seed=unified_seed + 44,
                                   quant_shift=patch_size if jitter else None, jitter=jitter, jit_seed=get_seed()))

    if full_equivariance:
        if flip_prob > 0:
            ret.append(RandomFlipTransform(prob_apply=flip_prob, random_seed=get_seed()))
        ret.append(_get_zoom(individual_scale, random_seed=get_seed()))

        ret.append(CenterCropTransform(image_size=crop_size))

        if max(perspective_scale) > 0:
            ret.append(RandomPerspectiveTransform(scale=perspective_scale, prob_apply=1.0, random_seed=get_seed()))

        if rot_max > 0:
            ret.append(RandomRotationTransform(abs_max=rot_max, random_seed=get_seed()))

    return ret


def get_pipeline(student_size: Union[int, Tuple[int, int]],
                 img_size: Union[int, Tuple[int, int]],
                 patch_size: int,
                 is_train: bool, is_teacher: bool,
                 max_img_size: int,
                 rng: np.random.Generator,
                 unified_seed: int,
                 full_equivariance: bool = False,
                 shift_equivariance: bool = False,
                 stochastic_size_args: Optional[Dict[str, Any]] = None,
                 stochastic_teacher: bool = False,
                 student_patch_size: Optional[int] = None,
                 scoped_rng: Optional[ScopedRNG] = None,
                 perf_test_simple_aug: bool = False,
                 ) -> CompositeTransform:
    """Build the transform chain for a single view.

    Args:
        student_size: Student model input size.
        img_size: Target image size for this view.
        patch_size: ViT patch size for this view.
        is_train: Whether the student is training.
        is_teacher: Whether this view is for a teacher.
        max_img_size: Maximum image dimension in the dataset.
        rng: Numpy RNG for seeding sub-transforms.
        unified_seed: Shared seed for transforms that must agree across views.
        full_equivariance: Enable full equivariant augmentation.
        shift_equivariance: Enable shift-equivariant jittered crops for teachers.
        stochastic_size_args: Dict with keys ``min_size``, ``max_size``,
            ``resolutions``, ``fixed_aspect`` for stochastic resolution.
        stochastic_teacher: Whether this teacher uses stochastic resolution.
        student_patch_size: Student's patch size (for teacher resolution rescaling).
        scoped_rng: Shared ``ScopedRNG`` for stochastic resolution.
        perf_test_simple_aug: Minimal resize + center-crop for benchmarking.

    Returns:
        A ``CompositeTransform`` that takes ``(DeferImage, Quad)`` tuples.
    """
    student_size: Tuple[int, int] = to_2tuple(student_size)
    img_size: Tuple[int, int] = to_2tuple(img_size)

    is_hi_res = max_img_size > 512

    jitter_size = patch_size

    transforms = []
    base_size = Size(*img_size)

    if perf_test_simple_aug:
        transforms = [
            MaxSizeTransform(image_size=base_size, smallest=True),
            CenterCropTransform(base_size),
        ]
        return CompositeTransform(transforms)

    if (is_train or stochastic_teacher) and stochastic_size_args:
        assert base_size.width == base_size.height
        patch_tx = None
        if student_patch_size != patch_size:
            patch_tx = lambda v: v * patch_size // student_patch_size
        crop_size = RandomSize(
            scoped_rng,
            base_size.height,
            min_size=stochastic_size_args.get('min_size', None),
            max_size=stochastic_size_args.get('max_size', None),
            step_size=patch_size,
            resolutions=stochastic_size_args.get('resolutions', None),
            fixed_aspect=stochastic_size_args.get('fixed_aspect', True),
            transform=patch_tx,
        )
    else:
        crop_size = base_size

    if not is_hi_res:
        transforms.append(MaxSizeTransform(image_size=crop_size, smallest=True))

        if is_train:
            transforms.extend(_get_deformations(crop_size,
                rng=rng,
                unified_seed=unified_seed,
                unified_scale=(1.0, 1.1),
                flip_prob=0.1,
                individual_scale=.95,
                rot_max=5.0,
                perspective_scale=(0.01, 0.05),
                patch_size=patch_size,
                jitter=False,
                full_equivariance=full_equivariance,
            ))
        elif is_teacher:
            transforms.extend(_get_deformations(crop_size,
                rng=rng,
                unified_seed=unified_seed,
                unified_scale=(1.0, 1.1),
                patch_size=jitter_size,
                jitter=shift_equivariance,
            ))
    else:
        transforms.append(MaxSizeTransform(image_size=crop_size, smallest=True))

        if is_train:
            transforms.extend(_get_deformations(crop_size,
                rng=rng,
                unified_seed=unified_seed,
                unified_scale=(0.9, 1.4),
                rot_max=1.0,
                perspective_scale=(0.01, 0.05),
                patch_size=patch_size,
                jitter=False,
                full_equivariance=full_equivariance,
            ))
        elif is_teacher:
            transforms.extend(_get_deformations(crop_size,
                rng=rng,
                unified_seed=unified_seed,
                unified_scale=(0.9, 1.4),
                patch_size=jitter_size,
                jitter=shift_equivariance,
            ))

    rand_pad = shift_equivariance and is_train and not is_teacher

    def get_seed():
        return rng.bit_generator.random_raw()

    if is_train and not is_teacher:
        crop = RandomCropTransform(image_size=crop_size, quant_shift=patch_size, random_seed=get_seed())
    else:
        crop = CenterCropTransform(crop_size)

    transforms.extend([
        PadToTransform(image_size=crop_size, quant_pad=patch_size, rand_pad=rand_pad),
        crop,
    ])

    transforms = CompositeTransform(transforms)

    return transforms
