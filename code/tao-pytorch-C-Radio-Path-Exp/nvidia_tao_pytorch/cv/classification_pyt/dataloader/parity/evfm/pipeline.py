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

"""Reference parity dumps for pipeline assembly helpers.

Imports resolve via PYTHONPATH pointing to the reference project root.
"""

import hashlib

import numpy as np
import torch
from PIL import Image

from nvidia_tao_pytorch.cv.classification_pyt.dataloader.parity import component


def _make_pil_image(seed=7, h=64, w=48):
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# -----------------------------------------------------------------------
# Listing helpers
# -----------------------------------------------------------------------

@component("listing_normalize")
def dump_listing_normalize():
    from data.data_pipe_v2 import _normalize_listing

    cases = {
        "comma_string": _normalize_listing("/a:0.3,/b:0.7"),
        "single_string": _normalize_listing("/only"),
        "list_of_strings": _normalize_listing(["/a:0.5", "/b"]),
        "list_of_tuples": _normalize_listing([("/a", "0.3"), ("/b", "0.7")]),
    }
    return {k: [[p, r] for p, r in v] for k, v in cases.items()}


@component("listing_simplify")
def dump_listing_simplify():
    from data.data_pipe_v2 import _simplify_listing

    cases = {
        "dedup": _simplify_listing([
            ("/a", 2.0), ("/b", 1.0), ("/a", 1.0),
        ]),
        "single": _simplify_listing([("/only", 3.0)]),
        "equal": _simplify_listing([("/x", 5.0), ("/y", 5.0)]),
    }
    return {k: [[p, r] for p, r in v] for k, v in cases.items()}


@component("listing_chain")
def dump_listing_chain():
    from data.data_pipe_v2 import _normalize_listing, _simplify_listing

    raw = "/a:0.3,/b:0.4,/a:0.3"
    normalized = _normalize_listing(raw)
    simplified = _simplify_listing(normalized)
    return {"result": [[p, r] for p, r in simplified]}


# -----------------------------------------------------------------------
# _sample_hook
# -----------------------------------------------------------------------

@component("sample_hook")
def dump_sample_hook():
    from data.data_pipe_v2 import _sample_hook

    stage = _sample_hook()

    fallback_img = _make_pil_image(seed=99, h=4, w=4)

    samples = [
        {"json": {"sha256": "sha_key"}, "jpg": b"img1"},
        {"json": {"md5": "md5_key"}, "jpg": b"img2"},
        {"json": {"uid": "uid_key"}, "jpg": b"img3"},
        {"json": {}, "jpg": fallback_img},
        {"jpg": fallback_img},
        {"json": {"sha256": "k", "caption": "cap_text"}, "jpg": b"img5"},
        {"json": {"sha256": "k", "caption": "ignored"}, "jpg": b"i", "txt": "existing"},
    ]

    results = list(stage(iter(samples)))

    return {
        "keys": [r["image_key"] for r in results],
        "txts": [r.get("txt", None) for r in results],
    }


# -----------------------------------------------------------------------
# _image_filter
# -----------------------------------------------------------------------

@component("image_filter")
def dump_image_filter():
    from data.data_pipe_v2 import _image_filter

    stage = _image_filter()

    samples = [
        {"jpg": b"1", "__key__": "s0"},
        {"png": b"2", "__key__": "s1"},
        {"txt": "hello", "__key__": "s2"},
        {"image.jpg": b"3", "__key__": "s3"},
        {"json": b"4", "__key__": "s4"},
        {"webp": b"5", "__key__": "s5"},
        {"data.bmp": b"6", "__key__": "s6"},
    ]

    results = list(stage(iter(samples)))
    return {"keys": [r["__key__"] for r in results]}


# -----------------------------------------------------------------------
# _prepare_image
# -----------------------------------------------------------------------

@component("prepare_image")
def dump_prepare_image():
    from data.data_pipe_v2 import _prepare_image

    pil_img = _make_pil_image(seed=42, h=64, w=48)

    stage = _prepare_image(num_replicas=2)
    samples = [(pil_img, "extra_field")]
    result = list(stage(iter(samples)))

    sample = result[0]

    replicas = []
    for i in range(2):
        di, q = sample[i]
        replicas.append({
            "img_sum": di.image.sum().item(),
            "img_shape": list(di.image.shape),
            "stm": di.stm.flatten().tolist(),
            "target_size": list(di.target_size),
            "bounds": q.bounds.flatten().tolist(),
        })

    return {
        "replicas": replicas,
        "extra": sample[2],
    }
