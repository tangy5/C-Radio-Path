# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""classification Data Module"""

import logging
from typing import Optional, Any, Dict, List
from torch.utils.data import DataLoader, distributed, RandomSampler, BatchSampler
import pytorch_lightning as pl
import torch

from nvidia_tao_pytorch.core.distributed.comm import is_dist_avail_and_initialized
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.dataset import CLDataset
from nvidia_tao_pytorch.cv.classification_pyt.dataloader import build_dataset


def _merge_stochastic_resolutions(per_teacher_dicts: List[Any]) -> Optional[Dict[int, float]]:
    """Merge per-teacher stochastic_resolutions: for each resolution take max prob
    across teachers, then renormalize. Returns a single dict or None if no teacher defines it.
    """
    merged: Dict[int, float] = {}
    for src in per_teacher_dicts:
        if src is None or not hasattr(src, 'items'):
            continue
        for res, prob in src.items():
            r = int(res)
            merged[r] = max(merged.get(r, 0.0), float(prob))
    if not merged:
        return None
    total = sum(merged.values())
    if total <= 0:
        return None
    return {r: p / total for r, p in merged.items()}


class CLDataModule(pl.LightningDataModule):
    """Lightning DataModule for Classification."""

    def __init__(self, dataset_config, experiment_config: Optional[Any] = None):
        """Lightning DataModule Initialization.

        Args:
            dataset_config: Configuration for the dataset.
            experiment_config: Full experiment config (optional). When
                provided and distillation with multi-view is used, the
                train dataset will be built with the equivariant pipeline.
        """
        super().__init__()
        self.dataset_config = dataset_config
        self.experiment_config = experiment_config
        self.batch_size = dataset_config["batch_size"]
        self.num_workers = dataset_config["workers"]
        self.root_dir = dataset_config["root_dir"]
        self.img_size = dataset_config["img_size"]
        self.augmentation = dataset_config["augmentation"]
        self.calib_dataset = None

        self.use_gpu_color_aug = self.augmentation.get("use_gpu_color_aug", False)
        if self.use_gpu_color_aug and not torch.cuda.is_available():
            logging.warning(
                "use_gpu_color_aug is True but CUDA is not available. "
                "GPU color augmentations will run on CPU, which may be slower."
            )

        self._train_loader = None
        self._shared_epoch = None
        self._loader_state = None

    def _build_wds_pipeline(self):
        """Build the equivariant WebDataset pipeline from experiment config.

        Translates TAO config fields into ``get_data_pipeline()`` arguments
        and returns the fully assembled loader, shared epoch, and loader
        state for checkpointing.
        """
        from nvidia_tao_pytorch.cv.classification_pyt.dataloader.data_pipeline import (
            PipelineConfig,
            get_data_pipeline,
        )
        from nvidia_tao_pytorch.cv.classification_pyt.dataloader.stages.utils import (
            seed_from_tuple,
        )

        # TODO(heslami): Move this to the dataset config with defaults values close to config def.
        train_cfg = self.dataset_config["train_dataset"]
        tar_sources = train_cfg["tar_data_sources"]

        ds_listing = [
            (src.get("root_dir", src.get("root")), src.get("scale_factor", 1.0))
            for src in tar_sources
        ]

        steps_per_epoch = sum(
            src.get("steps_per_epoch", 2000) for src in tar_sources
        )

        pipeline_config = PipelineConfig(
            steps_per_epoch=steps_per_epoch,
            workers=self.num_workers,
        )

        student_size = self.img_size
        student_patch_size = train_cfg.get("student_patch_size", 14)

        input_sizes = [student_size]
        patch_sizes = [student_patch_size]
        upsample_factors = []
        stochastic_teachers = []
        per_teacher_stochastic = []

        distill = getattr(self.experiment_config, "distill", None)
        teachers = []
        if distill is not None:
            teachers = getattr(distill, "teacher", [])
            if teachers and not hasattr(teachers, '__len__'):
                teachers = [teachers]

        for t in teachers:
            match_student = getattr(t, "match_student_resolution", True)
            teacher_input = getattr(t, "input_size", student_size)
            if match_student:
                input_sizes.append(student_size)
            else:
                input_sizes.append(teacher_input)
            patch_sizes.append(getattr(t, "patch_size", student_patch_size))
            upsample_factors.append(getattr(t, "upsample_factor", 1))
            stochastic_teachers.append(match_student)
            per_teacher_stochastic.append(
                getattr(t, "stochastic_resolutions", None)
            )

        stochastic_size_args = None
        merged_res = _merge_stochastic_resolutions(per_teacher_stochastic)
        if merged_res is not None:
            stochastic_size_args = {
                "resolutions": merged_res,
                "fixed_aspect": True,
            }

        base_seed = train_cfg.get("seed", 42)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        seed = seed_from_tuple(base_seed, 0, rank, 0)

        full_equivariance = train_cfg.get("full_equivariance", False)
        shift_equivariance = train_cfg.get("shift_equivariance", False)
        data_weight_mode = train_cfg.get("data_weight_mode", "inv_frequency")
        prefetch = train_cfg.get("prefetch", True)
        include_keys = train_cfg.get("include_keys", False)
        include_dataset_source = train_cfg.get("include_dataset_source", False)

        loader, shared_epoch, loader_state = get_data_pipeline(
            args=pipeline_config,
            ds_listing=ds_listing,
            input_sizes=input_sizes,
            patch_sizes=patch_sizes,
            batch_size=self.batch_size,
            is_train=True,
            epoch=0,
            seed=seed,
            upsample_factors=upsample_factors if upsample_factors else None,
            data_weight_mode=data_weight_mode,
            prefetch=prefetch,
            full_equivariance=full_equivariance,
            shift_equivariance=shift_equivariance,
            stochastic_size_args=stochastic_size_args,
            stochastic_teachers=stochastic_teachers if stochastic_teachers else None,
            include_keys=include_keys,
            include_dataset_source=include_dataset_source,
        )

        return loader, shared_epoch, loader_state

    def setup(self, stage: Optional[str] = None):
        """Setup the dataset.

        Args:
            stage: Stage of the dataset.
        """
        is_distributed = is_dist_avail_and_initialized()

        if stage == "fit" or stage is None:
            train_cfg = self.dataset_config["train_dataset"]
            tar_sources = train_cfg.get("tar_data_sources", [])
            has_tar = len(tar_sources) > 0

            if has_tar:
                self._train_loader, self._shared_epoch, self._loader_state = \
                    self._build_wds_pipeline()
                self.dataset = "WebDataset"
                self.train_sampler = None
            else:
                self.train_dataset = build_dataset(
                    images_dir=train_cfg["images_dir"],
                    augmentation=self.augmentation,
                    image_size=self.img_size,
                    root_dir=self.root_dir,
                )
                self.dataset = "CLDataset"
                if is_distributed:
                    self.train_sampler = distributed.DistributedSampler(
                        self.train_dataset, shuffle=True
                    )
                else:
                    self.train_sampler = RandomSampler(self.train_dataset)

            self.val_dataset = CLDataset(
                root_dir=self.root_dir,
                augmentation=self.augmentation,
                split="val",
                img_size=self.img_size,
                to_tensor=True,
                data_path=self.dataset_config["val_dataset"]["images_dir"],
            )

        if stage == "test" or stage is None:
            if self.dataset == "CLDataset":
                self.test_dataset = CLDataset(
                    root_dir=self.root_dir,
                    augmentation=self.augmentation,
                    split="val",
                    img_size=self.img_size,
                    to_tensor=True,
                    data_path=self.dataset_config["val_dataset"]["images_dir"],
                )
            else:
                raise NotImplementedError(
                    "Wrong dataset name %s (choose one from [CLDataset,])"
                    % self.dataset
                )

        if stage == "predict" or stage is None:
            if self.dataset == "CLDataset":
                self.predict_dataset = CLDataset(
                    root_dir=self.root_dir,
                    augmentation=self.augmentation,
                    split="test",
                    img_size=self.img_size,
                    to_tensor=True,
                    data_path=self.dataset_config["test_dataset"]["images_dir"],
                )
            else:
                raise NotImplementedError(
                    "Wrong dataset name %s (choose one from [CLDataset,])"
                    % self.dataset
                )

        if stage == "calibration" or stage is None:
            calib_cfg = self.dataset_config.get("quant_calibration_dataset", {})
            calib_images_dir = calib_cfg.get("images_dir", "") if hasattr(calib_cfg, 'get') else getattr(calib_cfg, "images_dir", "")
            if calib_images_dir:
                self.calib_dataset = CLDataset(
                    root_dir=self.root_dir,
                    augmentation=self.augmentation,
                    split="val",
                    img_size=self.img_size,
                    to_tensor=True,
                    data_path=calib_images_dir,
                )
            else:
                raise ValueError("quant_calibration_dataset.images_dir must be provided for calibration stage.")

    def train_dataloader(self):
        """Build the dataloader for training.

        Returns:
            train_loader: PyTorch DataLoader used for training.
        """
        if self.dataset == "WebDataset":
            return self._train_loader

        dataloader_kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": True,
            "persistent_workers": True,
            "drop_last": False,
        }
        dataloader_kwargs["batch_sampler"] = BatchSampler(
            self.train_sampler, self.batch_size, drop_last=True
        )
        dataloader_kwargs["collate_fn"] = self.train_dataset.collate_fn
        train_loader = DataLoader(self.train_dataset, **dataloader_kwargs)
        return train_loader

    def val_dataloader(self):
        """Build the dataloader for validation.

        Returns:
            val_loader: PyTorch DataLoader used for validation.
        """
        val_loader = DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.val_dataset.collate_fn,
            pin_memory=True,
        )
        return val_loader

    def test_dataloader(self):
        """Build the dataloader for evaluation.

        Returns:
            test_loader: PyTorch DataLoader used for evaluation.
        """
        test_loader = DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=False,
        )
        return test_loader

    def predict_dataloader(self):
        """Build the dataloader for inference.

        Returns:
            predict_loader: PyTorch DataLoader used for inference.
        """
        predict_loader = DataLoader(
            self.predict_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=False,
        )
        return predict_loader

    def calib_dataloader(self):
        """Build the dataloader for quantization calibration."""
        if self.calib_dataset is None:
            raise ValueError("Calibration dataset is not initialized. Please ensure quant_calibration_dataset.images_dir is set in the config.")
        calib_loader = DataLoader(
            self.calib_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=False,
            collate_fn=self.calib_dataset.collate_fn,
        )
        return calib_loader

    def set_epoch(self, epoch):
        """Set the current epoch for the dataset.

        For the WebDataset pipeline, updates the shared epoch counter
        used by ``MultiStreamShuffle`` for deterministic shard ordering.
        For regular datasets this is a no-op (Lightning handles epoch
        setting via the sampler).

        Args:
            epoch: The current epoch number.
        """
        if self._shared_epoch is not None:
            self._shared_epoch.set_value(epoch)

    @property
    def loader_state(self):
        """Access the ``LoaderState`` for checkpointing.

        Returns ``None`` when not using the equivariant WebDataset pipeline.
        """
        return self._loader_state
