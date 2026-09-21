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

"""Unit tests for filters and SampleWeightInjector."""

import math
import sqlite3

import pytest
from PIL import Image

from nvidia_tao_pytorch.cv.classification_pyt.dataloader.filters.uniform_color_filter import (
    UniformColorFilter,
)
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.filters.hash_count_filter import (
    get_hash_database,
    get_hash_count_filter,
)
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.stages.sample_weight_injector import (
    SampleWeightInjector,
)


# ---- UniformColorFilter ----

def _make_tuple_sample(img):
    """Wrap a PIL image in a tuple the way wds.to_tuple would."""
    return (img,)


@pytest.mark.cv_unit
def test_ucf_passes_varied_image():
    img = Image.new("RGB", (32, 32))
    img.putpixel((0, 0), (255, 0, 0))
    img.putpixel((1, 1), (0, 255, 0))

    ucf = UniformColorFilter(image_tuple_idx=0, threshold=1)
    results = list(ucf.run([_make_tuple_sample(img)]))
    assert len(results) == 1


@pytest.mark.cv_unit
def test_ucf_filters_solid_image():
    img = Image.new("RGB", (32, 32), color=(128, 128, 128))

    ucf = UniformColorFilter(image_tuple_idx=0, threshold=1)
    results = list(ucf.run([_make_tuple_sample(img)]))
    assert len(results) == 0
    assert ucf.num_filtered == 1
    assert ucf.num_seen == 1


@pytest.mark.cv_unit
def test_ucf_threshold_zero_passes_solid():
    img = Image.new("RGB", (32, 32), color=(128, 128, 128))

    ucf = UniformColorFilter(image_tuple_idx=0, threshold=0)
    results = list(ucf.run([_make_tuple_sample(img)]))
    assert len(results) == 0


@pytest.mark.cv_unit
def test_ucf_batch():
    varied = Image.new("RGB", (10, 10))
    varied.putpixel((0, 0), (255, 0, 0))
    solid = Image.new("RGB", (10, 10), color=(50, 50, 50))

    samples = [
        _make_tuple_sample(varied),
        _make_tuple_sample(solid),
        _make_tuple_sample(varied),
    ]

    ucf = UniformColorFilter(image_tuple_idx=0, threshold=1)
    results = list(ucf.run(samples))
    assert len(results) == 2
    assert ucf.num_filtered == 1


@pytest.mark.cv_unit
def test_ucf_l2_norm_logic():
    img = Image.new("RGB", (4, 4))
    img.putpixel((0, 0), (0, 0, 0))
    img.putpixel((1, 1), (3, 4, 0))

    extrema = img.getextrema()
    expected_range = math.sqrt(sum((e[1] - e[0]) ** 2 for e in extrema))
    assert expected_range == 5.0

    ucf = UniformColorFilter(image_tuple_idx=0, threshold=4)
    results = list(ucf.run([_make_tuple_sample(img)]))
    assert len(results) == 1

    ucf2 = UniformColorFilter(image_tuple_idx=0, threshold=6)
    results2 = list(ucf2.run([_make_tuple_sample(img)]))
    assert len(results2) == 0


# ---- hash_count_filter ----

def _create_test_db(db_path):
    """Create a minimal hash_counts.db for testing."""
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE md5_values (md5 BLOB, count INTEGER)")
    conn.execute("INSERT INTO md5_values VALUES (?, ?)", (bytes.fromhex("aabbccdd"), 5))
    conn.execute("INSERT INTO md5_values VALUES (?, ?)", (bytes.fromhex("11223344"), 200))
    conn.execute("INSERT INTO md5_values VALUES (?, ?)", (bytes.fromhex("deadbeef"), 50))
    conn.commit()
    conn.close()


@pytest.mark.cv_unit
def test_get_hash_database_found(tmp_path):
    _create_test_db(str(tmp_path / "hash_counts.db"))
    conn = get_hash_database(str(tmp_path))
    assert conn is not None


@pytest.mark.cv_unit
def test_get_hash_database_parent_search(tmp_path):
    _create_test_db(str(tmp_path / "hash_counts.db"))
    child = tmp_path / "sub" / "dir"
    child.mkdir(parents=True)
    conn = get_hash_database(str(child))
    assert conn is not None


@pytest.mark.cv_unit
def test_get_hash_database_not_found(tmp_path):
    deep = tmp_path / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    conn = get_hash_database(str(deep))
    assert conn is None


@pytest.mark.cv_unit
def test_get_hash_count_filter(tmp_path):
    db_path = str(tmp_path / "hash_counts.db")
    _create_test_db(db_path)
    conn = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False)
    conn.execute('PRAGMA query_only = 1')

    filter_set = get_hash_count_filter(conn, max_count=10)
    hexes = filter_set
    assert "11223344" in hexes
    assert "deadbeef" in hexes
    assert "aabbccdd" not in hexes


# ---- SampleWeightInjector ----

def _create_weight_db(db_path, entries):
    """Create a hash_counts.db with given (md5_bytes, count) entries."""
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE md5_values (md5 BLOB, count INTEGER)")
    for md5_bytes, count in entries:
        conn.execute("INSERT INTO md5_values VALUES (?, ?)", (md5_bytes, count))
    conn.commit()
    conn.close()


@pytest.mark.cv_unit
def test_swi_injects_weights(tmp_path):
    db_path = str(tmp_path / "hash_counts.db")
    _create_weight_db(db_path, [
        (b"key1", 10),
        (b"key2", 1),
        (b"key3", 100),
    ])

    conn = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False, uri=True)
    conn.execute('PRAGMA query_only = 1')

    def key_extractor(sample):
        return sample["_key"]

    swi = SampleWeightInjector(
        database=conn,
        key_extractor=key_extractor,
        weight_mode='inv_frequency',
        batch_size=3,
    )

    def source():
        yield {"_key": b"key1", "data": "a"}
        yield {"_key": b"key2", "data": "b"}
        yield {"_key": b"key3", "data": "c"}

    results = list(swi.run(source()))
    assert len(results) == 3
    assert results[0]["__weight__"] == pytest.approx(1.0 / 10)
    assert results[1]["__weight__"] == pytest.approx(1.0 / 1)
    assert results[2]["__weight__"] == pytest.approx(1.0 / 100)


@pytest.mark.cv_unit
def test_swi_unknown_key_gets_default_weight(tmp_path):
    db_path = str(tmp_path / "hash_counts.db")
    _create_weight_db(db_path, [(b"known", 50)])

    conn = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False, uri=True)
    conn.execute('PRAGMA query_only = 1')

    swi = SampleWeightInjector(
        database=conn,
        key_extractor=lambda s: s["_key"],
        weight_mode='inv_frequency',
        batch_size=2,
    )

    def source():
        yield {"_key": b"known", "data": "x"}
        yield {"_key": b"unknown", "data": "y"}

    results = list(swi.run(source()))
    assert len(results) == 2
    assert results[0]["__weight__"] == pytest.approx(1.0 / 50)
    assert results[1]["__weight__"] == pytest.approx(1.0 / 1)


@pytest.mark.cv_unit
@pytest.mark.parametrize("mode,count,expected", [
    ("inv_frequency", 4, 1.0 / 4),
    ("inv_sq_frequency", 4, 1.0 / 16),
    ("inv_log_frequency", 4, 1.0 / math.log(5)),
    ("uniform", 4, 1),
    ("none", 4, 1),
])
def test_swi_weight_modes(tmp_path, mode, count, expected):
    db_path = str(tmp_path / "hash_counts.db")
    _create_weight_db(db_path, [(b"k", count)])

    conn = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False, uri=True)
    conn.execute('PRAGMA query_only = 1')

    swi = SampleWeightInjector(
        database=conn,
        key_extractor=lambda s: s["_key"],
        weight_mode=mode,
        batch_size=1,
    )

    results = list(swi.run(iter([{"_key": b"k"}])))
    assert results[0]["__weight__"] == pytest.approx(expected)


@pytest.mark.cv_unit
def test_swi_invalid_mode():
    with pytest.raises(ValueError, match="Unsupported weight function"):
        SampleWeightInjector(
            database=sqlite3.connect(":memory:"),
            key_extractor=lambda s: s,
            weight_mode='bogus',
        )
