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

"""Unit tests for dataloader/stages/ components."""

import copy
import hashlib

import pytest

from nvidia_tao_pytorch.cv.classification_pyt.dataloader.stages.utils import (
    SharedEpoch,
    seed_from_tuple,
    compute_md5_hash,
    md5_str_to_bytes,
    iterator_exhauster,
    extract_dict_field,
    _SHARD_SHUFFLE_SIZE,
    _SHARD_SHUFFLE_INITIAL,
    _SAMPLE_SHUFFLE_SIZE,
    _SAMPLE_SHUFFLE_INITIAL,
)


# --- SharedEpoch ---

@pytest.mark.cv_unit
def test_shared_epoch_lifecycle():
    se = SharedEpoch(0)
    assert se.get_value() == 0
    se.set_value(5)
    assert se.get_value() == 5
    se.increment(3)
    assert se.get_value() == 8
    se.increment()
    assert se.get_value() == 9


# --- seed_from_tuple ---

@pytest.mark.cv_unit
def test_seed_from_tuple_stable():
    assert seed_from_tuple(42, 0, 0, 0) == seed_from_tuple(42, 0, 0, 0)


@pytest.mark.cv_unit
@pytest.mark.parametrize("args", [(0,), (42, 1, 2, 3), ("hello", 123)])
def test_seed_from_tuple_range(args):
    v = seed_from_tuple(*args)
    assert 0 <= v < 2**32


@pytest.mark.cv_unit
def test_seed_from_tuple_distinct():
    assert seed_from_tuple(1) != seed_from_tuple(2)


@pytest.mark.cv_unit
def test_seed_from_tuple_formula():
    ref = int(hashlib.sha256(repr((42,)).encode("utf-8")).hexdigest(), 16) % (2**32)
    assert seed_from_tuple(42) == ref


# --- compute_md5_hash ---

@pytest.mark.cv_unit
@pytest.mark.parametrize("value,expected_input", [
    ("hello", b"hello"),
    (b"world", b"world"),
    ("", b""),
])
def test_compute_md5_hash(value, expected_input):
    assert compute_md5_hash(value).hexdigest() == hashlib.md5(expected_input).hexdigest()


# --- md5_str_to_bytes ---

@pytest.mark.cv_unit
def test_md5_str_to_bytes_md5_key():
    fn = md5_str_to_bytes()
    md5_hex = hashlib.md5(b"test_sample").hexdigest()
    sample = {"json": f'{{"md5": "{md5_hex}", "other": "data"}}'.encode()}
    assert fn(sample) == bytes.fromhex(md5_hex)


@pytest.mark.cv_unit
def test_md5_str_to_bytes_sha256_key():
    fn = md5_str_to_bytes()
    sha_hex = hashlib.sha256(b"test2").hexdigest()
    sample = {"json": f'{{"sha256": "{sha_hex}"}}'.encode()}
    assert fn(sample) == bytes.fromhex(sha_hex)


# --- iterator_exhauster ---

@pytest.mark.cv_unit
def test_iterator_exhauster():
    call_ct = [0]

    def gen_fn():
        call_ct[0] += 1
        if call_ct[0] > 3:
            return None
        return iter(range(call_ct[0] * 10, call_ct[0] * 10 + 3))

    assert list(iterator_exhauster(gen_fn)) == [10, 11, 12, 20, 21, 22, 30, 31, 32]


# --- extract_dict_field ---

@pytest.mark.cv_unit
def test_extract_dict_field():
    data = [
        {"nested": {"key": "val1"}, "out": None},
        {"nested": {"key": "val2"}, "out": None},
    ]
    stage = extract_dict_field("nested.key", "out")
    results = list(stage(iter(copy.deepcopy(data))))
    assert [r["out"] for r in results] == ["val1", "val2"]
