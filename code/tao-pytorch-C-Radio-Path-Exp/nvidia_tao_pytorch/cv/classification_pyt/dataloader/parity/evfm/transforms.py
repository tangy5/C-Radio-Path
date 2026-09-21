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

"""Reference parity dumps for Phase 3: Quad, generate_homography_grid, DeferImage.

Imports resolve via PYTHONPATH pointing to the reference project root.
DeferImage uses the C++ ``spatial_transform`` kernel from ``adlr_ops_cpp``
(its default).
"""

import math

import torch

from nvidia_tao_pytorch.cv.classification_pyt.dataloader.parity import component


def _deterministic_image(C=3, H=32, W=48):
    """Create a reproducible float32 image tensor (same seed as TAO side)."""
    torch.manual_seed(0)
    return torch.rand(C, H, W)


def _standard_transforms(obj):
    """Apply a fixed sequence of transforms (must match TAO side exactly)."""
    snapshots = {}

    obj.translate(torch.tensor([5.0, 10.0]))
    snapshots["after_translate"] = _snapshot(obj)

    obj.scale(torch.tensor([2.0, 1.5]))
    snapshots["after_scale"] = _snapshot(obj)

    obj.flip(30.0)
    snapshots["after_flip"] = _snapshot(obj)

    angle = math.pi / 6
    rot = torch.tensor([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ], dtype=torch.float32)
    obj.rotate(rot)
    snapshots["after_rotate"] = _snapshot(obj)

    stm = torch.tensor([
        [1.0, 0.2, 0.0],
        [0.1, 1.0, 0.0],
        [3.0, -2.0, 1.0],
    ], dtype=torch.float32)
    obj.apply_stm(stm)
    snapshots["after_apply_stm"] = _snapshot(obj)

    return snapshots


def _snapshot(obj):
    from data.transforms.quad import Quad
    from data.transforms.defer_image import DeferImage

    if isinstance(obj, Quad):
        return obj.bounds.flatten().tolist()
    elif isinstance(obj, DeferImage):
        return {
            "stm": obj.stm.flatten().tolist(),
            "target_size": list(obj.target_size),
            "fresh": obj.fresh,
        }
    raise TypeError(type(obj))


@component("quad")
def dump_quad():
    from data.transforms.quad import Quad

    image = _deterministic_image()
    q = Quad(image)

    result = {"init_bounds": q.bounds.flatten().tolist()}
    result["transforms"] = _standard_transforms(q)
    result["final_bounds"] = q.bounds.flatten().tolist()

    return result


@component("homography_grid")
def dump_homography_grid():
    from data.transforms.generate_homography_grid import (
        generate_homography_grid,
        BASE_GRID_CACHE,
    )
    BASE_GRID_CACHE.clear()

    results = {}

    hom_identity = torch.eye(3).unsqueeze(0)
    g = generate_homography_grid(hom_identity, (1, 3, 8, 8))
    results["identity"] = {
        "shape": list(g.shape),
        "corners": [
            g[0, 0, 0].tolist(),
            g[0, 0, -1].tolist(),
            g[0, -1, 0].tolist(),
            g[0, -1, -1].tolist(),
        ],
        "sum": float(g.sum()),
    }

    hom_scale = torch.tensor([
        [0.5, 0, 0],
        [0, 2.0, 0],
        [0, 0, 1],
    ], dtype=torch.float32).unsqueeze(0)
    BASE_GRID_CACHE.clear()
    g = generate_homography_grid(hom_scale, (1, 3, 8, 8))
    results["scale"] = {
        "shape": list(g.shape),
        "corners": [
            g[0, 0, 0].tolist(),
            g[0, 0, -1].tolist(),
            g[0, -1, 0].tolist(),
            g[0, -1, -1].tolist(),
        ],
        "sum": float(g.sum()),
    }

    hom_translate = torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0.3, -0.2, 1],
    ], dtype=torch.float32).unsqueeze(0)
    BASE_GRID_CACHE.clear()
    g = generate_homography_grid(hom_translate, (1, 3, 16, 16))
    results["translate"] = {
        "shape": list(g.shape),
        "corners": [
            g[0, 0, 0].tolist(),
            g[0, 0, -1].tolist(),
            g[0, -1, 0].tolist(),
            g[0, -1, -1].tolist(),
        ],
        "sum": float(g.sum()),
    }

    N = 4
    hom_batch = torch.eye(3).unsqueeze(0).expand(N, -1, -1).contiguous()
    hom_batch = hom_batch.clone()
    hom_batch[1, 0, 0] = 0.5
    hom_batch[2, 1, 1] = 2.0
    hom_batch[3, 2, 0] = 0.1
    BASE_GRID_CACHE.clear()
    g = generate_homography_grid(hom_batch, (N, 3, 8, 8))
    results["batch"] = {
        "shape": list(g.shape),
        "per_batch_sums": [float(g[i].sum()) for i in range(N)],
    }

    return results


@component("defer_image_stm")
def dump_defer_image_stm():
    from data.transforms.defer_image import DeferImage

    image = _deterministic_image()
    di = DeferImage(image)

    result = {
        "init_stm": di.stm.flatten().tolist(),
        "init_target_size": list(di.target_size),
        "init_fresh": di.fresh,
    }

    result["transforms"] = _standard_transforms(di)

    result["final_stm"] = di.stm.flatten().tolist()
    result["final_target_size"] = list(di.target_size)
    result["final_fresh"] = di.fresh
    result["final_bds"] = di.get_bds().flatten().tolist()

    di2 = DeferImage(image)
    di2.clip_translate(4, 6, 20, 10)
    result["clip_translate"] = {
        "stm": di2.stm.flatten().tolist(),
        "target_size": list(di2.target_size),
    }

    di3 = DeferImage(image)
    di3.scale(torch.tensor([3.0, 2.0]), retain_dims=True)
    result["scale_retain_dims"] = {
        "stm": di3.stm.flatten().tolist(),
        "target_size": list(di3.target_size),
    }

    return result


@component("spatial_transform")
def dump_spatial_transform():
    from data.transforms.defer_image import DeferImage

    results = {}

    torch.manual_seed(42)
    image = torch.rand(3, 16, 16)

    di = DeferImage(image, allow_eager=False)
    out = di()
    results["identity_noop"] = {
        "shape": list(out.shape),
        "sum": float(out.sum()),
        "flat_head": out.flatten()[:20].tolist(),
    }

    di2 = DeferImage(image.clone())
    di2.translate(torch.tensor([3.0, 2.0]))
    out2 = di2()
    results["translate"] = {
        "shape": list(out2.shape),
        "sum": float(out2.sum()),
        "flat_head": out2.flatten()[:20].tolist(),
    }

    di3 = DeferImage(image.clone())
    di3.scale(torch.tensor([0.5, 0.5]))
    di3.target_size = tuple(int(x) for x in di3.target_size)
    out3 = di3()
    results["scale_down"] = {
        "shape": list(out3.shape),
        "sum": float(out3.sum()),
        "flat_head": out3.flatten()[:20].tolist(),
    }

    di4 = DeferImage(image.clone())
    di4.flip(8.0)
    out4 = di4()
    results["flip"] = {
        "shape": list(out4.shape),
        "sum": float(out4.sum()),
        "flat_head": out4.flatten()[:20].tolist(),
    }

    image_u8 = torch.randint(0, 256, (3, 8, 8), dtype=torch.uint8)
    di5 = DeferImage(image_u8.clone())
    di5.translate(torch.tensor([1.0, 1.0]))
    out5 = di5()
    results["uint8_translate"] = {
        "shape": list(out5.shape),
        "dtype": str(out5.dtype),
        "sum": float(out5.sum()),
        "flat_head": out5.flatten()[:20].tolist(),
    }

    return results
