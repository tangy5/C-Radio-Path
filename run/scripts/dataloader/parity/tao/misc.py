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

"""TAO parity dumps for dataloader/misc/ components."""

import random

import numpy as np
from PIL import Image

from nvidia_tao_pytorch.cv.classification_pyt.dataloader.parity import component


@component("fast_to_tensor")
def dump_fast_to_tensor():
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.ops.fast_to_tensor import (
        fast_to_tensor,
    )

    np.random.seed(0)
    imgs = {}
    for h, w in [(32, 32), (64, 48), (1, 1)]:
        np_img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        pil_img = Image.fromarray(np_img)
        t = fast_to_tensor(pil_img)
        imgs[f"{h}x{w}"] = {
            "shape": list(t.shape),
            "sum": float(t.sum()),
            "flat_head": t.flatten()[:20].tolist(),
        }
    return imgs


@component("mixture_emitter")
def dump_mixture_emitter():
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.misc.mixture_emitter import (
        MixtureEmitter,
    )

    results = {}
    for name, ks in [("single", [1.0]), ("two", [0.7, 0.3]), ("four", [1, 2, 3, 4])]:
        me = MixtureEmitter(ks)
        picks = [me.pick() for _ in range(500)]
        results[name] = {"picks": picks, "pick_ct": me.pick_ct}
    return results


@component("weighted_reservoir")
def dump_weighted_reservoir():
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.misc.weighted_streaming_sampler import (
        WeightedWSWoRReservoir,
    )

    R = 30
    seed = 42

    rng_build = random.Random(99)
    weights = [rng_build.random() + 0.01 for _ in range(100)]
    payloads = list(range(100))

    res = WeightedWSWoRReservoir(R, rng=random.Random(seed))
    res.fast_build(list(weights), list(payloads))

    drain = []
    while True:
        p, w = res.next_sample()
        if p is None:
            break
        drain.append({"payload": p, "weight": w})

    res2 = WeightedWSWoRReservoir(R, rng=random.Random(seed))
    init_rng = random.Random(77)
    init_w = [init_rng.random() + 0.01 for _ in range(R)]
    res2.fast_build(list(init_w), list(range(R)))

    gen_rng = random.Random(123)
    ctr = [0]

    def gen():
        ctr[0] += 1
        return gen_rng.random() + 0.01, f"s{ctr[0]}"

    refresh_out = [res2.get_next_and_refresh(gen) for _ in range(200)]

    return {
        "drain_payloads": [d["payload"] for d in drain],
        "drain_weights": [d["weight"] for d in drain],
        "refresh_out": refresh_out,
    }
