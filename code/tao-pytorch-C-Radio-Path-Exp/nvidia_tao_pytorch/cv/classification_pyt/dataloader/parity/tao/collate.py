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

"""Reference parity dump for equivariant_collate.

Extracts the same fields that the EVFM-side dump extracts so the
comparison is format-agnostic (EVFM outputs list-of-tuples, TAO outputs
dict — both dumps normalise to the same JSON structure).
"""

import torch

from nvidia_tao_pytorch.cv.classification_pyt.dataloader.parity import component


def _make_batch(num_views, batch_size=4, C=3, H=48, W=64):
    """Create a deterministic batch of (DeferImage, Quad) tuples."""
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.defer_image import DeferImage
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.quad import Quad

    batch = []
    for b in range(batch_size):
        torch.manual_seed(b * 17 + 3)
        views = []
        for _ in range(num_views):
            img = torch.rand(C, H, W)
            views.append((DeferImage(img), Quad(img)))
        batch.append(views)
    return batch


def _make_transformed_batch(batch_size=4, C=3, H=64, W=64):
    """Batch with different crops for student and teacher."""
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.defer_image import DeferImage
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.quad import Quad

    batch = []
    for b in range(batch_size):
        torch.manual_seed(b * 17 + 3)
        img = torch.rand(C, H, W)

        di_s = DeferImage(img.clone())
        q_s = Quad(img.clone())
        di_s.clip_translate(0, 0, 32, 32)
        q_s.translate(torch.tensor([0.0, 0.0]))

        di_t = DeferImage(img.clone())
        q_t = Quad(img.clone())
        di_t.clip_translate(16, 16, 32, 32)
        q_t.translate(torch.tensor([-16.0, -16.0]))

        batch.append([(di_s, q_s), (di_t, q_t)])
    return batch


def _extract_from_tao_dict(result, num_images):
    """Normalise TAO dict output to a flat JSON-friendly structure."""
    out = {}

    out["student_img_shape"] = list(result["img"].shape)
    out["student_img_sum"] = float(result["img"].sum())
    out["student_valid_mask_sum"] = float(result["valid_mask"].sum())
    out["student_valid_mask_shape"] = list(result["valid_mask"].shape)

    for t in range(num_images - 1):
        tv = result["teacher_views"][t]
        prefix = f"teacher_{t}"
        out[f"{prefix}_img_shape"] = list(tv["img"].shape)
        out[f"{prefix}_img_sum"] = float(tv["img"].sum())
        out[f"{prefix}_valid_mask_sum"] = float(tv["valid_mask"].sum())
        out[f"{prefix}_valid_mask_shape"] = list(tv["valid_mask"].shape)
        out[f"{prefix}_transform"] = tv["spatial_transform"].flatten().tolist()

    return out


@component("collate_identity")
def dump_collate_identity():
    """Identity transforms — student + 1 teacher, all fresh DeferImages."""
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.equivariant_collate import (
        equivariant_collate,
    )

    collate_fn = equivariant_collate(num_images=2, patch_sizes=[14, 14])
    batch = _make_batch(num_views=2, batch_size=4)
    result = collate_fn(batch)

    return _extract_from_tao_dict(result, num_images=2)


@component("collate_transformed")
def dump_collate_transformed():
    """Different crops for student and teacher."""
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.equivariant_collate import (
        equivariant_collate,
    )

    collate_fn = equivariant_collate(num_images=2, patch_sizes=[14, 14])
    batch = _make_transformed_batch(batch_size=4)
    result = collate_fn(batch)

    return _extract_from_tao_dict(result, num_images=2)


@component("collate_valid_mask")
def dump_collate_valid_mask():
    """Valid mask after crop — only a subset of pixels should be valid."""
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.defer_image import DeferImage
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.quad import Quad
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.equivariant_collate import (
        equivariant_collate,
    )

    collate_fn = equivariant_collate(num_images=1, patch_sizes=[14])

    torch.manual_seed(99)
    img = torch.rand(3, 48, 64)
    di = DeferImage(img)
    q = Quad(img)
    di.clip_translate(8, 8, 40, 32)
    q.translate(torch.tensor([-8.0, -8.0]))

    result = collate_fn([[(di, q)]])

    return {
        "valid_mask_shape": list(result["valid_mask"].shape),
        "valid_mask_sum": float(result["valid_mask"].sum()),
        "valid_mask_flat_head": result["valid_mask"].flatten()[:50].tolist(),
    }
