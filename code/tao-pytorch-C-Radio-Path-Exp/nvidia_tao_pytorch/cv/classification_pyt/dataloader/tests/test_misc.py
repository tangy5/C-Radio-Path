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

"""Unit tests for dataloader/misc/ components."""

import random

import numpy as np
import pytest
import torch
from PIL import Image

from nvidia_tao_pytorch.cv.classification_pyt.dataloader.ops.fast_to_tensor import (
    fast_to_tensor,
    fast_to_tensor_fallback,
    _cpp_fast_to_tensor,
)
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.misc.mixture_emitter import MixtureEmitter
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.misc.weighted_streaming_sampler import (
    WeightedWSWoRReservoir,
)


# --- fast_to_tensor ---

@pytest.mark.cv_unit
@pytest.mark.parametrize("h,w", [(64, 64), (224, 224), (1, 1), (100, 200)])
def test_fast_to_tensor_matches_torchvision(h, w):
    from torchvision.transforms.functional import to_tensor as tv_to_tensor

    np_img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    pil_img = Image.fromarray(np_img)

    ours = fast_to_tensor(pil_img)
    ref = tv_to_tensor(pil_img)

    assert ours.shape == (3, h, w)
    assert ours.dtype == torch.float32
    assert torch.allclose(ours, ref, rtol=0, atol=2e-7)


@pytest.mark.cv_unit
@pytest.mark.parametrize("h,w", [(64, 64), (224, 224), (1, 1), (100, 200)])
def test_fast_to_tensor_cpp_matches_fallback(h, w):
    """If C++ extension is available, compare it against the pure-Python path."""
    if _cpp_fast_to_tensor is None:
        pytest.skip("FastToTensor C++ extension not built")

    np_img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    pil_img = Image.fromarray(np_img)

    cpp_out = fast_to_tensor(pil_img)
    py_out = fast_to_tensor_fallback(pil_img)

    assert cpp_out.shape == py_out.shape
    assert torch.allclose(cpp_out, py_out, rtol=0, atol=2e-7)


@pytest.mark.cv_unit
def test_fast_to_tensor_rejects_non_image():
    with pytest.raises(TypeError):
        fast_to_tensor("not_an_image")


# --- MixtureEmitter ---

@pytest.mark.cv_unit
def test_mixture_emitter_single_source():
    me = MixtureEmitter([1.0])
    picks = [me.pick() for _ in range(50)]
    assert all(p == 0 for p in picks)


@pytest.mark.cv_unit
def test_mixture_emitter_proportions():
    me = MixtureEmitter([0.7, 0.3])
    picks = [me.pick() for _ in range(1000)]
    ratio = picks.count(0) / len(picks)
    assert abs(ratio - 0.7) < 0.02, f"Expected ~0.7, got {ratio:.3f}"
    assert sum(me.pick_ct) == 1000


@pytest.mark.cv_unit
def test_mixture_emitter_deterministic():
    a = MixtureEmitter([1, 2, 3])
    b = MixtureEmitter([1, 2, 3])
    assert [a.pick() for _ in range(200)] == [b.pick() for _ in range(200)]


# --- WeightedWSWoRReservoir ---

@pytest.mark.cv_unit
def test_reservoir_fast_build_and_drain():
    R = 20
    res = WeightedWSWoRReservoir(R, rng=random.Random(42))
    weights = [random.Random(99).random() + 0.01 for _ in range(100)]
    res.fast_build(weights, list(range(100)))
    assert res.live_count == 100

    drained = []
    while True:
        p, w = res.next_sample()
        if p is None:
            break
        drained.append(p)
    assert len(drained) == 100
    assert res.live_count == 0


@pytest.mark.cv_unit
def test_reservoir_deterministic():
    R = 20
    seed = 42
    weights = [random.Random(99).random() + 0.01 for _ in range(100)]
    payloads = list(range(100))

    r1 = WeightedWSWoRReservoir(R, rng=random.Random(seed))
    r2 = WeightedWSWoRReservoir(R, rng=random.Random(seed))
    r1.fast_build(list(weights), list(payloads))
    r2.fast_build(list(weights), list(payloads))

    d1, d2 = [], []
    while True:
        p1, _ = r1.next_sample()
        p2, _ = r2.next_sample()
        if p1 is None:
            break
        d1.append(p1)
        d2.append(p2)
    assert d1 == d2


@pytest.mark.cv_unit
def test_reservoir_refresh_maintains_size():
    R = 20
    res = WeightedWSWoRReservoir(R, rng=random.Random(42))
    init_w = [random.Random(77).random() + 0.01 for _ in range(R)]
    res.fast_build(init_w, list(range(R)))

    gen_rng = random.Random(123)
    ctr = [0]

    def gen():
        ctr[0] += 1
        return gen_rng.random() + 0.01, f"s{ctr[0]}"

    for _ in range(50):
        res.get_next_and_refresh(gen)
    assert res.live_count == R
