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

"""Unit tests for multi-source I/O stages (MultiStreamShuffle, MultiPipeSampler)."""

import io
import itertools
import tarfile

import pytest
import torch

from nvidia_tao_pytorch.cv.classification_pyt.dataloader.stages.utils import SharedEpoch
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.stages.multi_stream_shuffle import (
    MultiStreamShuffle,
)
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.stages.multi_pipe_sampler import (
    MultiPipeSampler,
)


def _make_wds_tar(path, sample_ids):
    """Create a minimal webdataset-format tar with the given sample IDs."""
    with tarfile.open(path, "w") as tf:
        for sid in sample_ids:
            name = f"{sid:06d}.txt"
            data = f"sample_{sid}".encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


@pytest.fixture
def wds_tar_dir(tmp_path):
    """Create a temp directory with 4 small webdataset tar files."""
    for shard_idx in range(4):
        shard_path = tmp_path / f"shard-{shard_idx:04d}.tar"
        sample_ids = list(range(shard_idx * 5, shard_idx * 5 + 5))
        _make_wds_tar(str(shard_path), sample_ids)
    return tmp_path


# ---- MultiStreamShuffle ----

@pytest.mark.cv_unit
def test_mss_basic_iteration(wds_tar_dir):
    urls = sorted(str(p) for p in wds_tar_dir.glob("*.tar"))
    mss = MultiStreamShuffle(urls=urls, seed=42, num_streams=1, epoch=0)
    samples = list(itertools.islice(mss, 20))
    assert len(samples) == 20
    for s in samples:
        assert "__key__" in s


@pytest.mark.cv_unit
def test_mss_visit_counter(wds_tar_dir):
    urls = sorted(str(p) for p in wds_tar_dir.glob("*.tar"))
    mss = MultiStreamShuffle(urls=urls, seed=42, num_streams=1, epoch=0)

    assert mss.visit_counter.sum().item() == 0

    n_samples = 20
    _ = list(itertools.islice(mss, n_samples))

    assert mss.visit_counter.sum().item() > 0
    assert all(c >= 0 for c in mss.visit_counter.tolist())


@pytest.mark.cv_unit
def test_mss_state_save_restore(wds_tar_dir):
    urls = sorted(str(p) for p in wds_tar_dir.glob("*.tar"))
    mss = MultiStreamShuffle(urls=urls, seed=42, num_streams=1, epoch=0)

    _ = list(itertools.islice(mss, 10))

    state = mss.get_state()
    assert state.shape == mss.visit_counter.shape
    assert state.sum().item() > 0

    mss2 = MultiStreamShuffle(urls=urls, seed=42, num_streams=1, epoch=0)
    assert mss2.visit_counter.sum().item() == 0
    mss2.load_state(state)
    assert torch.equal(mss2.visit_counter, state)


@pytest.mark.cv_unit
def test_mss_load_state_wrong_shape(wds_tar_dir):
    urls = sorted(str(p) for p in wds_tar_dir.glob("*.tar"))
    mss = MultiStreamShuffle(urls=urls, seed=42, num_streams=1, epoch=0)
    wrong_state = torch.zeros(100, dtype=torch.int64)
    with pytest.warns(UserWarning):
        result = mss.load_state(wrong_state)
    assert result is False


@pytest.mark.cv_unit
def test_mss_custom_reduce_urls(wds_tar_dir):
    urls = sorted(str(p) for p in wds_tar_dir.glob("*.tar"))

    def only_first_two(url_list, rng):
        return [(i, u) for i, u in enumerate(url_list) if i < 2]

    mss = MultiStreamShuffle(
        urls=urls, seed=42, num_streams=1, epoch=0,
        reduce_urls_fn=only_first_two,
    )
    _ = list(itertools.islice(mss, 10))

    assert mss.visit_counter[0].item() > 0
    assert mss.visit_counter[1].item() > 0
    assert mss.visit_counter[2].item() == 0
    assert mss.visit_counter[3].item() == 0


@pytest.mark.cv_unit
def test_mss_urls_sorted_on_init():
    urls = ["z.tar", "a.tar", "m.tar"]
    mss = MultiStreamShuffle(urls=urls, seed=0, num_streams=1, epoch=0)
    assert mss.urls == ["a.tar", "m.tar", "z.tar"]


@pytest.mark.cv_unit
def test_mss_shared_epoch(wds_tar_dir):
    urls = sorted(str(p) for p in wds_tar_dir.glob("*.tar"))
    se = SharedEpoch(5)
    mss = MultiStreamShuffle(urls=urls, seed=42, num_streams=1, epoch=se)
    _ = list(itertools.islice(mss, 5))
    assert se.get_value() == 5


# ---- MultiPipeSampler ----

def _make_pipe(prefix, n):
    """Create a simple iterable pipe yielding n dicts."""
    def pipe():
        for i in range(n):
            yield {"id": f"{prefix}_{i}", "__source__": prefix}
    return pipe()


@pytest.mark.cv_unit
def test_mps_single_pipe():
    pipes = [_make_pipe("A", 2000)]
    mps = MultiPipeSampler(pipes=pipes, rates=[1.0], epoch=0, seed=42, bufsize=200)
    samples = list(itertools.islice(mps, 50))
    assert len(samples) == 50
    assert all(s["__source__"] == "A" for s in samples)


@pytest.mark.cv_unit
def test_mps_weighted_mixing():
    pipes = [_make_pipe("A", 5000), _make_pipe("B", 5000)]
    mps = MultiPipeSampler(pipes=pipes, rates=[0.7, 0.3], epoch=0, seed=42, bufsize=200)
    samples = list(itertools.islice(mps, 1000))
    assert len(samples) == 1000

    a_count = sum(1 for s in samples if s["__source__"] == "A")
    ratio = a_count / len(samples)
    assert 0.55 < ratio < 0.85, f"Expected ~0.7, got {ratio:.3f}"


@pytest.mark.cv_unit
def test_mps_deterministic():
    def make_pipes():
        return [_make_pipe("A", 2000), _make_pipe("B", 2000)]

    mps1 = MultiPipeSampler(pipes=make_pipes(), rates=[0.6, 0.4], epoch=0, seed=99)
    mps2 = MultiPipeSampler(pipes=make_pipes(), rates=[0.6, 0.4], epoch=0, seed=99)

    s1 = [s["id"] for s in itertools.islice(mps1, 200)]
    s2 = [s["id"] for s in itertools.islice(mps2, 200)]
    assert s1 == s2


@pytest.mark.cv_unit
def test_mps_weight_key():
    def weighted_pipe():
        for i in range(2000):
            yield {"id": i, "__weight__": 2.0 + i * 0.01}

    pipes = [weighted_pipe()]
    mps = MultiPipeSampler(pipes=pipes, rates=[1.0], epoch=0, seed=42, bufsize=200)
    samples = list(itertools.islice(mps, 50))
    assert len(samples) == 50


@pytest.mark.cv_unit
def test_mps_three_sources():
    pipes = [_make_pipe("X", 3000), _make_pipe("Y", 3000), _make_pipe("Z", 3000)]
    mps = MultiPipeSampler(
        pipes=pipes, rates=[0.5, 0.3, 0.2], epoch=0, seed=42, bufsize=300,
    )
    samples = list(itertools.islice(mps, 500))
    assert len(samples) == 500

    counts = {}
    for s in samples:
        src = s["__source__"]
        counts[src] = counts.get(src, 0) + 1

    assert len(counts) == 3


@pytest.mark.cv_unit
def test_mps_shared_epoch():
    se = SharedEpoch(3)
    pipes = [_make_pipe("A", 2000)]
    mps = MultiPipeSampler(pipes=pipes, rates=[1.0], epoch=se, seed=42, bufsize=200)
    _ = list(itertools.islice(mps, 10))
    assert se.get_value() == 3
