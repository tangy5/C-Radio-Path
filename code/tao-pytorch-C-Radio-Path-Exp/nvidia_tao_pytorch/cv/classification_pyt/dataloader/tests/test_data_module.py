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

"""Unit tests for CLDataModule pipeline config translation."""

from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from nvidia_tao_pytorch.cv.classification_pyt.dataloader.pl_classification_data_module import (
    CLDataModule,
    _merge_stochastic_resolutions,
)


def _make_dataset_config(**overrides):
    """Build a minimal dataset_config dict for testing."""
    cfg = {
        "batch_size": 4,
        "workers": 2,
        "root_dir": "/data",
        "img_size": 224,
        "augmentation": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "train_dataset": {
            "tar_data_sources": [
                {"root_dir": "/data/shard_a", "scale_factor": 0.6, "steps_per_epoch": 500},
                {"root_dir": "/data/shard_b", "scale_factor": 0.4, "steps_per_epoch": 1000},
            ],
            "student_patch_size": 16,
            "seed": 99,
            "full_equivariance": True,
            "shift_equivariance": False,
            "data_weight_mode": "inv_frequency",
            "prefetch": True,
        },
        "val_dataset": {"images_dir": "/data/val"},
        "test_dataset": {"images_dir": "/data/test"},
    }
    cfg.update(overrides)
    return cfg


def _make_experiment_config(num_teachers=2, match_student=True, stochastic_res=None):
    """Build a minimal experiment_config with distill.teacher[] for testing."""
    teachers = []
    for i in range(num_teachers):
        t = SimpleNamespace(
            model=SimpleNamespace(backbone=SimpleNamespace(type=f"teacher_{i}")),
            input_size=256 + i * 32,
            patch_size=14,
            match_student_resolution=match_student,
            upsample_factor=1,
            stochastic_resolutions=stochastic_res,
        )
        teachers.append(t)
    return SimpleNamespace(distill=SimpleNamespace(teacher=teachers))


class TestMergeStochasticResolutions:

    def test_none_inputs(self):
        assert _merge_stochastic_resolutions([None, None]) is None

    def test_single_teacher(self):
        result = _merge_stochastic_resolutions([{224: 0.5, 256: 0.5}])
        assert result == {224: 0.5, 256: 0.5}

    def test_merge_takes_max_then_renormalizes(self):
        result = _merge_stochastic_resolutions([
            {224: 0.2, 256: 0.8},
            {224: 0.6, 256: 0.4},
        ])
        assert set(result.keys()) == {224, 256}
        assert abs(result[224] - 0.6 / (0.6 + 0.8)) < 1e-9
        assert abs(result[256] - 0.8 / (0.6 + 0.8)) < 1e-9

    def test_mixed_none_and_dict(self):
        result = _merge_stochastic_resolutions([None, {192: 0.1, 224: 0.9}])
        assert result is not None
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_empty_list(self):
        assert _merge_stochastic_resolutions([]) is None


class TestBuildWdsPipeline:
    """Test that _build_wds_pipeline() translates config correctly.

    We mock get_data_pipeline to capture the arguments it receives
    without needing actual tar files or a running distributed setup.
    """

    @patch("nvidia_tao_pytorch.cv.classification_pyt.dataloader.pl_classification_data_module.torch.distributed")
    @patch("nvidia_tao_pytorch.cv.classification_pyt.dataloader.data_pipeline.get_data_pipeline")
    def test_basic_two_teachers_match_student(self, mock_gdp, mock_dist):
        mock_dist.is_initialized.return_value = False
        mock_gdp.return_value = (MagicMock(), MagicMock(), MagicMock())

        dataset_cfg = _make_dataset_config()
        exp_cfg = _make_experiment_config(num_teachers=2, match_student=True)
        dm = CLDataModule(dataset_cfg, experiment_config=exp_cfg)
        dm._build_wds_pipeline()

        mock_gdp.assert_called_once()
        kwargs = mock_gdp.call_args
        call_kwargs = kwargs.kwargs if kwargs.kwargs else {}
        if not call_kwargs:
            call_kwargs = dict(zip(
                ["args", "ds_listing", "input_sizes", "patch_sizes",
                 "batch_size", "is_train", "epoch", "seed"],
                kwargs.args
            ))
            call_kwargs.update(kwargs.kwargs or {})

        assert call_kwargs["ds_listing"] == [
            ("/data/shard_a", 0.6),
            ("/data/shard_b", 0.4),
        ]

        assert call_kwargs["input_sizes"] == [224, 224, 224]
        assert call_kwargs["patch_sizes"] == [16, 14, 14]
        assert call_kwargs["batch_size"] == 4
        assert call_kwargs["is_train"] is True
        assert call_kwargs["epoch"] == 0
        assert call_kwargs["full_equivariance"] is True
        assert call_kwargs["shift_equivariance"] is False
        assert call_kwargs["data_weight_mode"] == "inv_frequency"
        assert call_kwargs["prefetch"] is True

        assert call_kwargs["args"].steps_per_epoch == 1000
        assert call_kwargs["args"].workers == 2

    @patch("nvidia_tao_pytorch.cv.classification_pyt.dataloader.pl_classification_data_module.torch.distributed")
    @patch("nvidia_tao_pytorch.cv.classification_pyt.dataloader.data_pipeline.get_data_pipeline")
    def test_teacher_own_resolution(self, mock_gdp, mock_dist):
        mock_dist.is_initialized.return_value = False
        mock_gdp.return_value = (MagicMock(), MagicMock(), MagicMock())

        dataset_cfg = _make_dataset_config()
        exp_cfg = _make_experiment_config(num_teachers=2, match_student=False)
        dm = CLDataModule(dataset_cfg, experiment_config=exp_cfg)
        dm._build_wds_pipeline()

        call_kwargs = mock_gdp.call_args.kwargs
        assert call_kwargs["input_sizes"] == [224, 256, 288]

    @patch("nvidia_tao_pytorch.cv.classification_pyt.dataloader.pl_classification_data_module.torch.distributed")
    @patch("nvidia_tao_pytorch.cv.classification_pyt.dataloader.data_pipeline.get_data_pipeline")
    def test_stochastic_resolutions_merged(self, mock_gdp, mock_dist):
        mock_dist.is_initialized.return_value = False
        mock_gdp.return_value = (MagicMock(), MagicMock(), MagicMock())

        dataset_cfg = _make_dataset_config()
        exp_cfg = _make_experiment_config(
            num_teachers=2,
            match_student=True,
            stochastic_res={224: 0.3, 256: 0.7},
        )
        dm = CLDataModule(dataset_cfg, experiment_config=exp_cfg)
        dm._build_wds_pipeline()

        call_kwargs = mock_gdp.call_args.kwargs
        ssa = call_kwargs["stochastic_size_args"]
        assert ssa is not None
        assert "resolutions" in ssa
        assert ssa["fixed_aspect"] is True
        assert abs(sum(ssa["resolutions"].values()) - 1.0) < 1e-9

    @patch("nvidia_tao_pytorch.cv.classification_pyt.dataloader.pl_classification_data_module.torch.distributed")
    @patch("nvidia_tao_pytorch.cv.classification_pyt.dataloader.data_pipeline.get_data_pipeline")
    def test_no_teachers(self, mock_gdp, mock_dist):
        mock_dist.is_initialized.return_value = False
        mock_gdp.return_value = (MagicMock(), MagicMock(), MagicMock())

        dataset_cfg = _make_dataset_config()
        exp_cfg = SimpleNamespace()
        dm = CLDataModule(dataset_cfg, experiment_config=exp_cfg)
        dm._build_wds_pipeline()

        call_kwargs = mock_gdp.call_args.kwargs
        assert call_kwargs["input_sizes"] == [224]
        assert call_kwargs["patch_sizes"] == [16]
        assert call_kwargs["upsample_factors"] is None
        assert call_kwargs["stochastic_teachers"] is None
        assert call_kwargs["stochastic_size_args"] is None

    @patch("nvidia_tao_pytorch.cv.classification_pyt.dataloader.pl_classification_data_module.torch.distributed")
    @patch("nvidia_tao_pytorch.cv.classification_pyt.dataloader.data_pipeline.get_data_pipeline")
    def test_steps_per_epoch_takes_max(self, mock_gdp, mock_dist):
        mock_dist.is_initialized.return_value = False
        mock_gdp.return_value = (MagicMock(), MagicMock(), MagicMock())

        dataset_cfg = _make_dataset_config()
        dataset_cfg["train_dataset"]["tar_data_sources"] = [
            {"root_dir": "/a", "scale_factor": 1.0, "steps_per_epoch": 200},
            {"root_dir": "/b", "scale_factor": 1.0, "steps_per_epoch": 800},
            {"root_dir": "/c", "scale_factor": 1.0, "steps_per_epoch": 500},
        ]
        exp_cfg = SimpleNamespace()
        dm = CLDataModule(dataset_cfg, experiment_config=exp_cfg)
        dm._build_wds_pipeline()

        assert mock_gdp.call_args.kwargs["args"].steps_per_epoch == 800


class TestSetEpoch:

    def test_set_epoch_updates_shared_epoch(self):
        dataset_cfg = _make_dataset_config()
        dm = CLDataModule(dataset_cfg)
        mock_se = MagicMock()
        dm._shared_epoch = mock_se
        dm.set_epoch(5)
        mock_se.set_value.assert_called_once_with(5)

    def test_set_epoch_noop_without_shared_epoch(self):
        dataset_cfg = _make_dataset_config()
        dm = CLDataModule(dataset_cfg)
        dm.set_epoch(5)


class TestLoaderStateProperty:

    def test_returns_none_by_default(self):
        dataset_cfg = _make_dataset_config()
        dm = CLDataModule(dataset_cfg)
        assert dm.loader_state is None

    def test_returns_loader_state_when_set(self):
        dataset_cfg = _make_dataset_config()
        dm = CLDataModule(dataset_cfg)
        sentinel = object()
        dm._loader_state = sentinel
        assert dm.loader_state is sentinel
