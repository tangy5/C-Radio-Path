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

"""TAO parity dumps for dataloader/stages/ components."""

import hashlib
import io
import itertools
import os
import tarfile
import tempfile

from nvidia_tao_pytorch.cv.classification_pyt.dataloader.parity import component


@component("seed_from_tuple")
def dump_seed_from_tuple():
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.stages.utils import (
        seed_from_tuple,
    )

    cases = [(42,), (42, 0, 0, 0), (42, 1, 2, 3), (0,), (999, 7, 13)]
    return {repr(c): seed_from_tuple(*c) for c in cases}


@component("md5_str_to_bytes")
def dump_md5_str_to_bytes():
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.stages.utils import (
        md5_str_to_bytes,
    )

    fn = md5_str_to_bytes()

    md5_hex = hashlib.md5(b"test_sample").hexdigest()
    sample1 = {"json": f'{{"md5": "{md5_hex}", "other": "data"}}'.encode()}

    sha_hex = hashlib.sha256(b"test2").hexdigest()
    sample2 = {"json": f'{{"sha256": "{sha_hex}"}}'.encode()}

    return {"md5_key": fn(sample1).hex(), "sha256_key": fn(sample2).hex()}


@component("shared_epoch")
def dump_shared_epoch():
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.stages.utils import SharedEpoch

    se = SharedEpoch(0)
    trace = [se.get_value()]
    se.set_value(5)
    trace.append(se.get_value())
    se.increment(3)
    trace.append(se.get_value())
    se.increment()
    trace.append(se.get_value())
    return trace


@component("iterator_exhauster")
def dump_iterator_exhauster():
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.stages.utils import (
        iterator_exhauster,
    )

    call_ct = [0]

    def gen_fn():
        call_ct[0] += 1
        if call_ct[0] > 3:
            return None
        return iter(range(call_ct[0] * 10, call_ct[0] * 10 + 3))

    return list(iterator_exhauster(gen_fn))


@component("compute_md5_hash")
def dump_compute_md5_hash():
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.stages.utils import (
        compute_md5_hash,
    )

    return {
        "hello": compute_md5_hash("hello").hexdigest(),
        "world_bytes": compute_md5_hash(b"world").hexdigest(),
        "empty": compute_md5_hash("").hexdigest(),
    }


def _make_wds_tar(path, sample_ids):
    """Create a minimal webdataset-format tar for parity testing."""
    with tarfile.open(path, "w") as tf:
        for sid in sample_ids:
            name = f"{sid:06d}.txt"
            data = f"sample_{sid}".encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


@component("multi_stream_shuffle")
def dump_multi_stream_shuffle():
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.stages.multi_stream_shuffle import (
        MultiStreamShuffle,
    )

    tmp_dir = tempfile.mkdtemp()
    try:
        for shard_idx in range(4):
            path = os.path.join(tmp_dir, f"shard-{shard_idx:04d}.tar")
            _make_wds_tar(path, list(range(shard_idx * 3, shard_idx * 3 + 3)))

        urls = sorted(
            os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.endswith(".tar")
        )
        mss = MultiStreamShuffle(urls=urls, seed=42, num_streams=1, epoch=0)

        keys = []
        for s in itertools.islice(mss, 24):
            keys.append(s["__key__"])

        return {
            "sample_keys": keys,
            "visit_counter": mss.visit_counter.tolist(),
        }
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


@component("multi_pipe_sampler")
def dump_multi_pipe_sampler():
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.stages.multi_pipe_sampler import (
        MultiPipeSampler,
    )

    def make_pipe(prefix, n):
        def pipe():
            for i in range(n):
                yield {"id": f"{prefix}_{i}", "__weight__": 1.0 + (i % 5) * 0.2}
        return pipe()

    pipes = [make_pipe("A", 2000), make_pipe("B", 2000)]
    mps = MultiPipeSampler(
        pipes=pipes, rates=[0.6, 0.4], epoch=0, seed=99, bufsize=200,
    )

    ids = []
    for s in itertools.islice(mps, 300):
        ids.append(s["id"])

    a_count = sum(1 for x in ids if x.startswith("A_"))
    b_count = sum(1 for x in ids if x.startswith("B_"))

    return {
        "sample_ids_head": ids[:50],
        "sample_ids_tail": ids[-50:],
        "total": len(ids),
        "a_count": a_count,
        "b_count": b_count,
    }


@component("sample_weight_injector")
def dump_sample_weight_injector():
    import sqlite3
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.stages.sample_weight_injector import (
        SampleWeightInjector,
    )

    db_path = os.path.join(tempfile.mkdtemp(), "hash_counts.db")
    try:
        conn_w = sqlite3.connect(db_path)
        conn_w.execute("CREATE TABLE md5_values (md5 BLOB, count INTEGER)")
        for md5_hex, count in [("aabb", 1), ("ccdd", 10), ("eeff", 100), ("1122", 50)]:
            conn_w.execute("INSERT INTO md5_values VALUES (?, ?)", (bytes.fromhex(md5_hex), count))
        conn_w.commit()
        conn_w.close()

        conn = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False, uri=True)
        conn.execute('PRAGMA query_only = 1')

        swi = SampleWeightInjector(
            database=conn,
            key_extractor=lambda s: s["_key"],
            weight_mode='inv_frequency',
            batch_size=4,
        )

        def source():
            for hex_key in ["aabb", "ccdd", "eeff", "1122"]:
                yield {"_key": bytes.fromhex(hex_key), "hex": hex_key}

        results = list(itertools.islice(swi.run(source()), 4))
        return {
            s["hex"]: s["__weight__"]
            for s in results
        }
    finally:
        import shutil
        shutil.rmtree(os.path.dirname(db_path), ignore_errors=True)


@component("uniform_color_filter")
def dump_uniform_color_filter():
    import math
    from PIL import Image
    from nvidia_tao_pytorch.cv.classification_pyt.dataloader.filters.uniform_color_filter import (
        UniformColorFilter,
    )

    solid = Image.new("RGB", (16, 16), color=(100, 100, 100))
    varied = Image.new("RGB", (16, 16), color=(0, 0, 0))
    varied.putpixel((0, 0), (255, 128, 64))
    near_solid = Image.new("RGB", (16, 16), color=(50, 50, 50))
    near_solid.putpixel((0, 0), (51, 50, 50))

    samples = [(solid,), (varied,), (near_solid,)]

    ucf = UniformColorFilter(image_tuple_idx=0, threshold=2)
    kept = list(ucf.run(iter(samples)))

    extrema_varied = varied.getextrema()
    range_varied = math.sqrt(sum((e[1] - e[0]) ** 2 for e in extrema_varied))

    extrema_near_solid = near_solid.getextrema()
    range_near_solid = math.sqrt(sum((e[1] - e[0]) ** 2 for e in extrema_near_solid))

    return {
        "num_kept": len(kept),
        "num_filtered": ucf.num_filtered,
        "num_seen": ucf.num_seen,
        "range_varied": range_varied,
        "range_near_solid": range_near_solid,
    }
