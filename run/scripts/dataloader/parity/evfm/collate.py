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

"""Reference parity dumps for equivariant_collate.

Imports resolve via PYTHONPATH pointing to the reference project root.
Extracts the same fields as the TAO side for format-agnostic comparison.
"""

import torch

from nvidia_tao_pytorch.cv.classification_pyt.dataloader.parity import component


def _make_batch(num_views, batch_size=4, C=3, H=48, W=64):
    """Create the same deterministic batch as the TAO side."""
    from data.transforms.defer_image import DeferImage
    from data.transforms.quad import Quad

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
    from data.transforms.defer_image import DeferImage
    from data.transforms.quad import Quad

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


def _extract_from_evfm_list(groups, num_images):
    """Normalise EVFM list-of-tuples output to the same JSON structure.

    EVFM format::

        groups[0] = (student_images, student_valid_masks)
        groups[i] = (teacher_images, teacher_valid_masks, transforms)  # i>0
        groups[-1] = extras_dict
    """
    out = {}

    student_imgs, student_vmasks = groups[0]
    out["student_img_shape"] = list(student_imgs.shape)
    out["student_img_sum"] = float(student_imgs.sum())
    out["student_valid_mask_sum"] = float(student_vmasks.sum())
    out["student_valid_mask_shape"] = list(student_vmasks.shape)

    for t in range(1, num_images):
        teacher_imgs, teacher_vmasks, teacher_transforms = groups[t]
        prefix = f"teacher_{t - 1}"
        out[f"{prefix}_img_shape"] = list(teacher_imgs.shape)
        out[f"{prefix}_img_sum"] = float(teacher_imgs.sum())
        out[f"{prefix}_valid_mask_sum"] = float(teacher_vmasks.sum())
        out[f"{prefix}_valid_mask_shape"] = list(teacher_vmasks.shape)
        out[f"{prefix}_transform"] = teacher_transforms.flatten().tolist()

    return out


@component("collate_identity")
def dump_collate_identity():
    """Identity transforms — student + 1 teacher, all fresh DeferImages."""
    from data.transforms.equivariant_collate import equivariant_collate

    collate_fn = equivariant_collate(num_images=2, patch_sizes=[14, 14])
    batch = _make_batch(num_views=2, batch_size=4)
    groups = collate_fn(batch)

    return _extract_from_evfm_list(groups, num_images=2)


@component("collate_transformed")
def dump_collate_transformed():
    """Different crops for student and teacher."""
    from data.transforms.equivariant_collate import equivariant_collate

    collate_fn = equivariant_collate(num_images=2, patch_sizes=[14, 14])
    batch = _make_transformed_batch(batch_size=4)
    groups = collate_fn(batch)

    return _extract_from_evfm_list(groups, num_images=2)


@component("collate_valid_mask")
def dump_collate_valid_mask():
    """Valid mask after crop — only a subset of pixels should be valid."""
    from data.transforms.defer_image import DeferImage
    from data.transforms.quad import Quad
    from data.transforms.equivariant_collate import equivariant_collate

    collate_fn = equivariant_collate(num_images=1, patch_sizes=[14])

    torch.manual_seed(99)
    img = torch.rand(3, 48, 64)
    di = DeferImage(img)
    q = Quad(img)
    di.clip_translate(8, 8, 40, 32)
    q.translate(torch.tensor([-8.0, -8.0]))

    groups = collate_fn([[(di, q)]])

    student_imgs, student_vmasks = groups[0]

    return {
        "valid_mask_shape": list(student_vmasks.shape),
        "valid_mask_sum": float(student_vmasks.sum()),
        "valid_mask_flat_head": student_vmasks.flatten()[:50].tolist(),
    }
