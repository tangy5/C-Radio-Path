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

"""Unit tests for data_pipeline.py helper functions."""

import hashlib
import io

import numpy as np
import pytest
import torch
from PIL import Image

from nvidia_tao_pytorch.cv.classification_pyt.dataloader.data_pipeline import (
    LoaderState,
    PipelineConfig,
    _image_filter,
    _img_format_handler,
    _normalize_listing,
    _prepare_image,
    _remove_pkl,
    _reset_rng,
    _sample_hook,
    _simplify_listing,
    _source_enricher,
)
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.base import ScopedRNG
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.defer_image import DeferImage
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.transforms.quad import Quad


# ---------------------------------------------------------------------------
# _normalize_listing
# ---------------------------------------------------------------------------

@pytest.mark.cv_unit
def test_normalize_listing_comma_string():
    result = _normalize_listing("/data/a:0.3,/data/b:0.7")
    assert result == [["/data/a", 0.3], ["/data/b", 0.7]]


@pytest.mark.cv_unit
def test_normalize_listing_single_path_string():
    result = _normalize_listing("/data/a")
    assert result == [["/data/a", 1]]


@pytest.mark.cv_unit
def test_normalize_listing_list_of_strings():
    result = _normalize_listing(["/data/a:0.5", "/data/b"])
    assert result == [["/data/a", 0.5], ["/data/b", 1]]


@pytest.mark.cv_unit
def test_normalize_listing_list_of_tuples():
    result = _normalize_listing([("/data/a", "0.3"), ("/data/b", "0.7")])
    assert result == [["/data/a", 0.3], ["/data/b", 0.7]]


@pytest.mark.cv_unit
def test_normalize_listing_does_not_mutate_input():
    original = ["/data/a:0.5", "/data/b:0.5"]
    copy = list(original)
    _normalize_listing(original)
    assert original == copy


# ---------------------------------------------------------------------------
# _simplify_listing
# ---------------------------------------------------------------------------

@pytest.mark.cv_unit
def test_simplify_listing_dedup():
    listing = [("/data/a", 2.0), ("/data/b", 1.0), ("/data/a", 1.0)]
    result = _simplify_listing(listing)
    result_dict = dict(result)
    assert abs(result_dict["/data/a"] - 0.75) < 1e-9
    assert abs(result_dict["/data/b"] - 0.25) < 1e-9


@pytest.mark.cv_unit
def test_simplify_listing_normalizes():
    listing = [("/data/a", 5.0), ("/data/b", 5.0)]
    result = _simplify_listing(listing)
    result_dict = dict(result)
    assert abs(result_dict["/data/a"] - 0.5) < 1e-9
    assert abs(result_dict["/data/b"] - 0.5) < 1e-9


@pytest.mark.cv_unit
def test_simplify_listing_single():
    listing = [("/data/only", 3.0)]
    result = _simplify_listing(listing)
    assert result == [("/data/only", 1.0)]


# ---------------------------------------------------------------------------
# _remove_pkl
# ---------------------------------------------------------------------------

@pytest.mark.cv_unit
def test_remove_pkl_strips_key():
    samples = [{"jpg": b"img", "pkl": b"pickle_data", "__key__": "0"}]
    result = list(_remove_pkl(iter(samples)))
    assert len(result) == 1
    assert "pkl" not in result[0]
    assert result[0]["jpg"] == b"img"


@pytest.mark.cv_unit
def test_remove_pkl_no_pkl_passthrough():
    samples = [{"jpg": b"img", "__key__": "0"}]
    result = list(_remove_pkl(iter(samples)))
    assert result == samples


# ---------------------------------------------------------------------------
# _source_enricher
# ---------------------------------------------------------------------------

@pytest.mark.cv_unit
def test_source_enricher():
    stage = _source_enricher("/data/my_dataset")
    samples = [{"jpg": b"img"}, {"jpg": b"img2"}]
    result = list(stage(iter(samples)))
    assert all(s["__dataset__"] == "/data/my_dataset" for s in result)


# ---------------------------------------------------------------------------
# _sample_hook
# ---------------------------------------------------------------------------

@pytest.mark.cv_unit
def test_sample_hook_sha256():
    stage = _sample_hook()
    samples = [{"json": {"sha256": "abc123"}, "jpg": b"img"}]
    result = list(stage(iter(samples)))
    assert result[0]["image_key"] == "abc123"


@pytest.mark.cv_unit
def test_sample_hook_md5_fallback():
    stage = _sample_hook()
    samples = [{"json": {"md5": "md5hash"}, "jpg": b"img"}]
    result = list(stage(iter(samples)))
    assert result[0]["image_key"] == "md5hash"


@pytest.mark.cv_unit
def test_sample_hook_uid_fallback():
    stage = _sample_hook()
    samples = [{"json": {"uid": "uid_val"}, "jpg": b"img"}]
    result = list(stage(iter(samples)))
    assert result[0]["image_key"] == "uid_val"


@pytest.mark.cv_unit
def test_sample_hook_hash_from_image():
    pil_img = Image.fromarray(
        np.zeros((4, 4, 3), dtype=np.uint8)
    )
    expected = hashlib.sha256(pil_img.tobytes()).hexdigest()
    stage = _sample_hook()
    samples = [{"json": {}, "jpg": pil_img}]
    result = list(stage(iter(samples)))
    assert result[0]["image_key"] == expected


@pytest.mark.cv_unit
def test_sample_hook_no_json():
    pil_img = Image.fromarray(
        np.ones((4, 4, 3), dtype=np.uint8)
    )
    expected = hashlib.sha256(pil_img.tobytes()).hexdigest()
    stage = _sample_hook()
    samples = [{"jpg": pil_img}]
    result = list(stage(iter(samples)))
    assert result[0]["image_key"] == expected


@pytest.mark.cv_unit
def test_sample_hook_caption_extraction():
    stage = _sample_hook()
    samples = [{"json": {"sha256": "key", "caption": "hello world"}, "jpg": b"i"}]
    result = list(stage(iter(samples)))
    assert result[0]["txt"] == "hello world"


@pytest.mark.cv_unit
def test_sample_hook_no_caption_if_txt_exists():
    stage = _sample_hook()
    samples = [
        {"json": {"sha256": "key", "caption": "from_json"}, "jpg": b"i", "txt": "original"}
    ]
    result = list(stage(iter(samples)))
    assert result[0]["txt"] == "original"


# ---------------------------------------------------------------------------
# _image_filter
# ---------------------------------------------------------------------------

@pytest.mark.cv_unit
def test_image_filter_keeps_jpg():
    stage = _image_filter()
    samples = [{"jpg": b"img", "__key__": "0"}]
    result = list(stage(iter(samples)))
    assert len(result) == 1


@pytest.mark.cv_unit
def test_image_filter_keeps_png():
    stage = _image_filter()
    samples = [{"png": b"img", "__key__": "0"}]
    result = list(stage(iter(samples)))
    assert len(result) == 1


@pytest.mark.cv_unit
def test_image_filter_drops_non_image():
    stage = _image_filter()
    samples = [{"txt": "hello", "__key__": "0"}]
    result = list(stage(iter(samples)))
    assert len(result) == 0


@pytest.mark.cv_unit
def test_image_filter_dotted_extension():
    stage = _image_filter()
    samples = [{"image.jpg": b"img", "__key__": "0"}]
    result = list(stage(iter(samples)))
    assert len(result) == 1
    assert "jpg" in result[0]


# ---------------------------------------------------------------------------
# _img_format_handler
# ---------------------------------------------------------------------------

@pytest.mark.cv_unit
def test_img_format_handler_pil_passthrough():
    pil_img = Image.fromarray(
        np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    )
    stage = _img_format_handler()
    samples = [(pil_img, "extra")]
    result = list(stage(iter(samples)))
    assert len(result) == 1
    assert isinstance(result[0][0], Image.Image)
    assert result[0][1] == "extra"


@pytest.mark.cv_unit
def test_img_format_handler_bytes_to_pil():
    pil_img = Image.fromarray(
        np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    )
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    stage = _img_format_handler()
    samples = [(img_bytes,)]
    result = list(stage(iter(samples)))
    assert len(result) == 1
    assert isinstance(result[0][0], Image.Image)


@pytest.mark.cv_unit
def test_img_format_handler_bad_bytes_skipped():
    stage = _img_format_handler(error_handler=lambda e: True)
    samples = [(b"not_an_image",)]
    result = list(stage(iter(samples)))
    assert len(result) == 0


# ---------------------------------------------------------------------------
# _prepare_image
# ---------------------------------------------------------------------------

@pytest.mark.cv_unit
def test_prepare_image_structure():
    pil_img = Image.fromarray(
        np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    )
    stage = _prepare_image(num_replicas=2)
    samples = [(pil_img, "extra_field")]
    result = list(stage(iter(samples)))
    assert len(result) == 1
    sample = result[0]
    assert isinstance(sample[0], tuple)
    assert isinstance(sample[0][0], DeferImage)
    assert isinstance(sample[0][1], Quad)
    assert isinstance(sample[1], tuple)
    assert isinstance(sample[1][0], DeferImage)
    assert isinstance(sample[1][1], Quad)
    assert sample[2] == "extra_field"


@pytest.mark.cv_unit
def test_prepare_image_replicas_independent():
    pil_img = Image.fromarray(
        np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    )
    stage = _prepare_image(num_replicas=3)
    samples = [(pil_img,)]
    result = list(stage(iter(samples)))
    sample = result[0]
    assert len(sample) == 3
    di0 = sample[0][0]
    di1 = sample[1][0]
    assert torch.equal(di0.image, di1.image)


# ---------------------------------------------------------------------------
# _reset_rng
# ---------------------------------------------------------------------------

@pytest.mark.cv_unit
def test_reset_rng_resets_at_batch_boundary():
    scoped = ScopedRNG(42)
    batch_size = 3
    stage = _reset_rng(scoped, batch_size)

    # Force seed computation so _rng_seed is non-None
    _ = scoped.seed
    assert scoped._rng_seed is not None

    data = list(range(7))
    result = list(stage(iter(data)))
    assert result == data
    # After processing 7 items with batch_size=3, reset_seed was called
    # at item 3 and item 6, leaving _rng_seed as None
    assert scoped._rng_seed is None


@pytest.mark.cv_unit
def test_reset_rng_no_reset_within_batch():
    scoped = ScopedRNG(42)
    batch_size = 100
    stage = _reset_rng(scoped, batch_size)

    # Force seed computation
    initial_seed = scoped.seed
    assert scoped._rng_seed is not None

    data = list(range(5))
    list(stage(iter(data)))
    # No reset happened (5 < 100), seed unchanged
    assert scoped._rng_seed == initial_seed


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------

@pytest.mark.cv_unit
def test_pipeline_config_defaults():
    cfg = PipelineConfig()
    assert cfg.steps_per_epoch == 2000
    assert cfg.workers == 8


@pytest.mark.cv_unit
def test_pipeline_config_custom():
    cfg = PipelineConfig(steps_per_epoch=500, workers=4)
    assert cfg.steps_per_epoch == 500
    assert cfg.workers == 4


# ---------------------------------------------------------------------------
# LoaderState (single-rank, no dist)
# ---------------------------------------------------------------------------

@pytest.mark.cv_unit
def test_loader_state_empty():
    state = LoaderState([])
    sd = state.state_dict()
    assert sd == [[]]


@pytest.mark.cv_unit
def test_loader_state_round_trip():
    counter = torch.zeros(5, dtype=torch.int64).share_memory_()
    counter[2] = 10

    class FakeStateObj:
        def __init__(self, c):
            self.c = c

        def get_state(self):
            return self.c.clone()

        def load_state(self, s):
            self.c.copy_(s)

    obj = FakeStateObj(counter)
    state = LoaderState([obj])
    sd = state.state_dict()

    counter.zero_()
    assert counter[2].item() == 0

    state.restore(sd)
    assert counter[2].item() == 10
