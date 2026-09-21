# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Distiller module for classification model"""
import os
import logging
import re
import random
import copy
from typing import Sequence
import numpy as np

import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torch.nn as nn

import torch.optim as optim
from torch.optim import lr_scheduler
from torchmetrics.classification import Accuracy
from torchmetrics import MetricCollection
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from transformers.optimization import get_cosine_schedule_with_warmup

import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.callbacks.loggers import TAOStatusLogger
from nvidia_tao_pytorch.core.callbacks.ema import EMA, EMAModelCheckpoint
from nvidia_tao_pytorch.core.utilities import get_latest_checkpoint

from nvidia_tao_pytorch.core.distillation.distiller import Distiller

from timm.data.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from nvidia_tao_pytorch.cv.classification_pyt.distillation.loss import DistillationLoss
from nvidia_tao_pytorch.cv.classification_pyt.distillation.validate import (
    build_knn_index,
    knn_eval_batch,
)
from nvidia_tao_pytorch.cv.classification_pyt.distillation.vitdet import (
    VitDetArgs,
    apply_vitdet_to_vit,
)
from nvidia_tao_pytorch.cv.classification_pyt.distillation.featsharp_adaptor import (
    wrap_teacher_with_featsharp,
)
from nvidia_tao_pytorch.cv.classification_pyt.model.classifier import build_model
from nvidia_tao_pytorch.cv.classification_pyt.utils.loss import Cross_Entropy
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.dataset import NOCLASS_IDX
logger = logging.getLogger(__name__)


class ClassDistiller(Distiller):
    """Classification Distiller"""

    def __init__(self, experiment_spec, export=False):
        """Initializes the distiller from given experiment_spec."""
        # Init local params
        self.experiment_spec = experiment_spec
        self.checkpoint_filename = "classifier_model"
        self.dataset_config = self.experiment_spec.dataset
        self.model_config = self.experiment_spec.model
        self.train_config = self.experiment_spec.train
        self.eval_config = self.experiment_spec.evaluate
        self.infer_config = self.experiment_spec.inference
        self.distill_config = self.experiment_spec.distill

        self.status_logging_dict = {}
        self.lr = self.train_config.optim.lr
        self.optimizer = self.train_config.optim
        self.lr_policy = self.optimizer.policy
        self.lr_policy_params = self.optimizer.policy_params
        self.max_epochs = self.train_config.num_epochs
        self.monitor_name = self.train_config.optim.monitor_name

        self.num_classes = self.dataset_config.num_classes
        
        # Parse teacher configurations (support single or multiple teachers)
        self.teacher_configs = self._parse_teacher_configs()
        
        # Global defaults for backward compatibility
        self.distill_weight = self.distill_config.loss_lambda
        self.distill_loss = self.distill_config.loss_type
        if self.distill_loss == "FD" or self.distill_loss == "CS":
            assert self.num_classes == 0, "Number of classes must be 0 when using `FD` or `CS` as the distillation loss type"

        # construct prediction id 2 class name mapping for visualization
        self.id_2_class_names = {}
        self.class_names = []
        with open(os.path.join(self.dataset_config.root_dir, "classes.txt")) as f:
            for idx, line in enumerate(f):
                self.id_2_class_names[idx] = line.strip()
                self.class_names.append(line.strip())

        # #  training log
        self.epoch_acc = 0
        self.max_num_epochs = self.train_config.num_epochs
        self.batch = None
        self.vis_dir = self.experiment_spec.results_dir
        self.optimizer_G = None

        self.vis_after_n_batches = self.eval_config.vis_after_n_batches
        self.vis_after_n_batches_infer = self.infer_config.vis_after_n_batches
        # init the model
        super().__init__(experiment_spec, export)

        train_acc = {}
        val_acc = {}
        if self.num_classes > 0:
            for topk in self.model_config.head.topk:
                train_acc[f"train_acc_{topk}"] = Accuracy(
                    task="multiclass",
                    num_classes=self.num_classes,
                    top_k=topk,
                    ignore_index=NOCLASS_IDX,
                )
                val_acc[f"val_acc_{topk}"] = Accuracy(
                    task="multiclass",
                    num_classes=self.num_classes,
                    top_k=topk,
                    ignore_index=NOCLASS_IDX,
                )
        self.train_acc = MetricCollection(train_acc)
        self.valid_acc = MetricCollection(val_acc)
        self.batch_size = self.dataset_config.batch_size

        self.register_buffer(
            "_student_mean", torch.tensor(OPENAI_CLIP_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_student_std", torch.tensor(OPENAI_CLIP_STD).view(1, 3, 1, 1)
        )

    def configure_callbacks(self) -> Sequence[Callback] | pl.Callback:
        """Configures logging and checkpoint-saving callbacks"""
        # This is called when trainer.fit() is called
        self.checkpoint_filename = "classifier_model"
        callbacks = []
        results_dir = self.experiment_spec["results_dir"]

        status_logger_callback = TAOStatusLogger(
            results_dir,
            append=True,
        )

        resume_ckpt = self.experiment_spec["train"][
            "resume_training_checkpoint_path"
        ] or get_latest_checkpoint(results_dir)

        resumed_epoch = 0
        if resume_ckpt:
            resumed_epoch = re.search("epoch_(\\d+)", resume_ckpt)
            if resumed_epoch is not None:
                resumed_epoch = int(resumed_epoch.group(1))
            else:
                resumed_epoch = 0

        status_logger_callback.epoch_counter = resumed_epoch + 1
        callbacks.append(status_logger_callback)

        if self.experiment_spec["train"]["enable_ema"]:
            # Apply Exponential Moving Average Callback
            ema_callback = EMA(**self.experiment_spec["train"]["ema"])
            ckpt_func = EMAModelCheckpoint
            callbacks.append(ema_callback)
        else:
            ckpt_func = ModelCheckpoint

        ModelCheckpoint.FILE_EXTENSION = ".pth"
        ModelCheckpoint.CHECKPOINT_EQUALS_CHAR = "_"

        if not self.checkpoint_filename:
            raise NotImplementedError(
                "checkpoint_filename not set in __init__() of model"
            )
        ModelCheckpoint.CHECKPOINT_NAME_LAST = f"{self.checkpoint_filename}_latest"

        # --- Milestone checkpoint (epoch-based, rotating last N) ---
        milestone_keep = self.experiment_spec["train"].get("milestone_keep", 7)
        milestone_cb = ckpt_func(
            every_n_epochs=1,                # Save every epoch
            dirpath=results_dir,
            save_on_train_epoch_end=True,
            monitor="epoch",         # Always increases → keeps last N
            mode="max",
            save_top_k=milestone_keep,       # Rotate through N slots
            save_last=False,
            filename="milestone_{epoch:03d}",
            enable_version_counter=False,
        )
        callbacks.append(milestone_cb)

        # --- Latest checkpoint (step-based, replace for SLURM resume) ---
        latest_interval = self.experiment_spec["train"].get("latest_checkpoint_interval", None)
        if latest_interval is not None:
            latest_cb = ckpt_func(
                every_n_train_steps=latest_interval,
                dirpath=results_dir,
                save_on_train_epoch_end=False,
                monitor="step",
                mode="max",
                save_top_k=3,                # Keep 3 latest for resume
                save_last="link",           # Symlink for easy resume
                filename="latest_{epoch:03d}_{step:06d}",
                enable_version_counter=False,
            )
            callbacks.append(latest_cb)

        return callbacks

    def _parse_teacher_configs(self):
        """Parse teacher configurations from config.
        
        Returns:
            List of dicts, each containing:
                - 'model_config': ModelConfig for the teacher
                - 'loss_type': Loss type for this teacher (str)
                - 'loss_lambda': Weight for this teacher (float)
                - 'pretrained_path': Path to pretrained model (str)
                - 'mode': Distillation mode (str)
        """
        teacher_cfg = self.distill_config.teacher

        # Check if teacher is a list (multiple teachers)
        if isinstance(teacher_cfg, (list, tuple)):
            teacher_list = teacher_cfg
        else:
            teacher_list = [teacher_cfg]
        
        # Handle case where teacher_list[0] is itself a list (nested list from schema)
        import omegaconf
        if len(teacher_list) == 1 and isinstance(teacher_list[0], (list, tuple, omegaconf.ListConfig)):
            logger.info(f"Unwrapping nested teacher list")
            teacher_list = list(teacher_list[0])
        
        parsed_configs = []
        logger.info(f"Processing {len(teacher_list)} teachers")
        for idx, teacher in enumerate(teacher_list):
            config = {}
            
            # Debug logging
            logger.info(f"_parse_teacher_configs: Teacher {idx} type: {type(teacher)}")
            logger.info(f"_parse_teacher_configs: Teacher {idx} has 'model': {hasattr(teacher, 'model')}")
            if hasattr(teacher, 'model'):
                logger.info(f"_parse_teacher_configs: Teacher {idx} model type: {type(teacher.model)}")
            
            # Check if this is a TeacherConfig (with model, loss_type, loss_lambda fields)
            # or a plain ModelConfig
            if hasattr(teacher, 'model') and not isinstance(teacher.model, (list, tuple)):
                # This is a TeacherConfig
                config['model_config'] = teacher.model
                config['loss_type'] = teacher.loss_type if teacher.loss_type is not None else self.distill_config.loss_type
                config['loss_lambda'] = teacher.loss_lambda if teacher.loss_lambda is not None else self.distill_config.loss_lambda
                config['pretrained_path'] = getattr(teacher, 'pretrained_teacher_model_path', None)
                config['mode'] = getattr(teacher, 'mode', self.distill_config.mode or 'auto')
            else:
                # This is a plain ModelConfig - use global settings
                config['model_config'] = teacher
                config['loss_type'] = self.distill_config.loss_type
                config['loss_lambda'] = self.distill_config.loss_lambda
                config['pretrained_path'] = getattr(self.distill_config, 'pretrained_teacher_model_path', None)
                config['mode'] = self.distill_config.mode or 'auto'

            # Multi-view: per-teacher input_size, match_student_resolution, stochastic_resolutions (EVFM-style)
            config['input_size'] = getattr(teacher, 'input_size', None)
            config['match_student_resolution'] = getattr(teacher, 'match_student_resolution', True)
            config['stochastic_resolutions'] = getattr(teacher, 'stochastic_resolutions', None)
            # Per-teacher image normalization (e.g. [0.5, 0.5, 0.5] for SAM3/SigLIP2; ImageNet for DINOv3)
            _nm = getattr(teacher, 'norm_mean', None)
            _ns = getattr(teacher, 'norm_std', None)
            config['norm_mean'] = list(_nm) if _nm and len(_nm) == 3 else None
            config['norm_std'] = list(_ns) if _ns and len(_ns) == 3 else None
            config['summary_loss_weight'] = getattr(teacher, 'summary_loss_weight', 1.0)
            config['fd_loss_weight'] = getattr(teacher, 'fd_loss_weight', 1.0)
            config['summary_loss_type'] = getattr(teacher, 'summary_loss_type', 'CE')
            # FeatSharp adaptor
            config['adaptor'] = getattr(teacher, 'adaptor', None)
            config['upsampler_checkpoint'] = getattr(teacher, 'upsampler_checkpoint', None)
            config['do_upsample'] = getattr(teacher, 'do_upsample', True)
            config['featsharp_lib_path'] = getattr(teacher, 'featsharp_lib_path', None)

            parsed_configs.append(config)
            logger.info(f"Teacher {idx}: loss_type={config['loss_type']}, "
                       f"loss_lambda={config['loss_lambda']}, mode={config['mode']}")
        
        return parsed_configs

    @staticmethod
    def _normalize_input(
        x: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize an image tensor from [0,1] using the given mean/std buffers."""
        return (x - mean) / std

    def _normalize_student_input(self, img: torch.Tensor) -> torch.Tensor:
        """Normalize student input from [0,1] using OPENAI_CLIP statistics."""
        return self._normalize_input(img, self._student_mean, self._student_std)

    def _apply_teacher_normalization(
        self, teacher_input: torch.Tensor, teacher_config: dict, device: torch.device
    ) -> torch.Tensor:
        """Normalize teacher input from [0,1] to per-teacher normalization.

        All dataloaders now output [0,1] images, so normalization is always
        ``(x - mean_t) / std_t``, matching EVFM's ``InputConditioner``.
        """
        norm_mean = teacher_config.get("norm_mean")
        norm_std = teacher_config.get("norm_std")
        if not norm_mean or not norm_std:
            logger.warning(
                "Teacher has no norm_mean/norm_std configured -- "
                "passing raw [0,1] input which is likely incorrect. "
                "Add norm_mean/norm_std to the teacher config."
            )
            return teacher_input

        mean_t = torch.tensor(norm_mean, dtype=teacher_input.dtype, device=device).view(1, 3, 1, 1)
        std_t = torch.tensor(norm_std, dtype=teacher_input.dtype, device=device).view(1, 3, 1, 1)
        return self._normalize_input(teacher_input, mean_t, std_t)

    def _setup_bindings(self):
        """Setup bindings to be captured during training for distillation."""
        pass

    def _build_model(self, export=False):
        """Internal function to build the model."""
        # Build multiple teacher models
        self.teachers = nn.ModuleList()
        import omegaconf

        for idx, teacher_config in enumerate(self.teacher_configs):
            # Get the model config from teacher - this is a ModelConfig with backbone and head
            model_cfg = teacher_config['model_config']
            
            # Directly extract backbone and head from model_cfg without full conversion
            backbone_dict = {}
            head_dict = None
            
            # Get backbone config
            if isinstance(model_cfg, omegaconf.DictConfig):
                # Access backbone directly as attribute
                if 'backbone' in model_cfg:
                    backbone_cfg = model_cfg.backbone
                    # Convert just the backbone to container
                    if isinstance(backbone_cfg, omegaconf.DictConfig):
                        backbone_dict = omegaconf.OmegaConf.to_container(backbone_cfg, resolve=True)
                    elif hasattr(backbone_cfg, '__dataclass_fields__'):
                        import dataclasses
                        backbone_dict = dataclasses.asdict(backbone_cfg)
                # Get head
                if 'head' in model_cfg and model_cfg.head is not None:
                    head_cfg = model_cfg.head
                    if isinstance(head_cfg, omegaconf.DictConfig):
                        head_dict = omegaconf.OmegaConf.to_container(head_cfg, resolve=True)
                    elif hasattr(head_cfg, '__dataclass_fields__'):
                        import dataclasses
                        head_dict = dataclasses.asdict(head_cfg)
            elif hasattr(model_cfg, '__dataclass_fields__'):
                # It's a dataclass - convert backbone and head separately
                import dataclasses
                if hasattr(model_cfg, 'backbone') and model_cfg.backbone:
                    backbone_dict = dataclasses.asdict(model_cfg.backbone)
                if hasattr(model_cfg, 'head') and model_cfg.head:
                    head_dict = dataclasses.asdict(model_cfg.head)
            elif isinstance(model_cfg, dict):
                if 'backbone' in model_cfg and model_cfg['backbone']:
                    b = model_cfg['backbone']
                    if isinstance(b, dict):
                        backbone_dict = dict(b)
                    elif hasattr(b, '__dataclass_fields__'):
                        import dataclasses
                        backbone_dict = dataclasses.asdict(b)
                if 'head' in model_cfg and model_cfg['head']:
                    h = model_cfg['head']
                    if isinstance(h, dict):
                        head_dict = dict(h)
                    elif hasattr(h, '__dataclass_fields__'):
                        import dataclasses
                        head_dict = dataclasses.asdict(h)
            
            # Set pretrained path if available
            if teacher_config['pretrained_path'] is not None:
                backbone_dict['pretrained_backbone_path'] = teacher_config['pretrained_path']
            
            # Debug logging
            logger.info(f"Teacher {idx} model_cfg type: {type(model_cfg)}")
            logger.info(f"Teacher {idx} model_cfg value: {model_cfg}")
            logger.info(f"Teacher {idx} backbone_dict: {backbone_dict}")
            
            # Get backbone type for validation
            backbone_type = backbone_dict.get('type', '')
            if 'radio' in backbone_type:
                assert self.num_classes == 0, f"Number of classes must be 0 when using radio as teacher {idx}"
            
            loss_type = teacher_config['loss_type']
            if loss_type in ["FD", "CS", "balanced", "MSE"]:
                assert self.num_classes == 0, \
                    f"Number of classes must be 0 when using `{loss_type}` for teacher {idx}"
            
            # Create minimal experiment config as plain dict - completely bypass schema
            dataset_dict = omegaconf.OmegaConf.to_container(self.experiment_spec.dataset, resolve=True)
            teacher_cfg_dict = {
                'model': {
                    'backbone': backbone_dict,
                    'head': head_dict
                },
                'dataset': dataset_dict
            }
            
            # Create new plain config without any schema
            teacher_cfg = omegaconf.OmegaConf.create(teacher_cfg_dict)
            
            # Build the teacher model
            teacher_model = build_model(experiment_config=teacher_cfg, export=export)
            teacher_model.eval()
            
            # Freeze teacher
            for _, param in teacher_model.named_parameters():
                param.requires_grad = False
            
            for module in teacher_model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
                if isinstance(module, nn.LayerNorm):
                    module.eval()
                if isinstance(module, nn.Dropout):
                    module.eval()
            
            self.teachers.append(teacher_model)
            logger.info(f"Built teacher model {idx}: {backbone_type}")

        # Build the student model
        self.model = build_model(experiment_config=self.experiment_spec, export=export)
        self.model.train()

        # Apply LoRA if configured (freeze backbone, inject low-rank adapters)
        lora_config = getattr(self.experiment_spec.model.backbone, 'lora', None)
        if lora_config and getattr(lora_config, 'enable', False):
            from nvidia_tao_pytorch.cv.backbone_v2.lora import (
                apply_lora_to_radio,
                freeze_non_lora_params,
                print_lora_summary,
            )
            apply_lora_to_radio(self.model, lora_config)
            freeze_non_lora_params(self.model)
            print_lora_summary(self.model)
        
        # For backward compatibility, keep single teacher reference if only one teacher
        if len(self.teachers) == 1:
            self.teacher = self.teachers[0]

    def _build_criterion(self):
        """Internal function to build the loss function."""
        assert self.model_config.head.loss.type in [
            "CrossEntropyLoss"
        ], "Only CrossEntropyLoss is supported."
        if self.model_config.head.loss.type == "CrossEntropyLoss":
            self.criterion = Cross_Entropy(
                label_smoothing=self.model_config.head.loss.label_smooth_val,
            )
        else:
            raise NotImplementedError(self.train_config["loss"])

        # Create multiple distillation loss modules, one per teacher
        self.distillation_loss_fns = nn.ModuleList()
        
        for idx, (teacher_model, teacher_config) in enumerate(zip(self.teachers, self.teacher_configs)):
            loss_fn = DistillationLoss(
                loss_type=teacher_config['loss_type'],
                student_model=self.model,
                teacher_model=teacher_model,
                distillation_mode=teacher_config['mode'],
                num_classes=self.num_classes,
                temperature=getattr(self.distill_config, 'temperature', 1.0),
                use_mlp=getattr(self.distill_config, 'use_mlp', True),
                mlp_hidden_size=getattr(self.distill_config, 'mlp_hidden_size', 1024),
                mlp_num_inner=getattr(self.distill_config, 'mlp_num_inner', 0),
                summary_loss_weight=teacher_config.get('summary_loss_weight', 1.0),
                fd_loss_weight=teacher_config.get('fd_loss_weight', 1.0),
                summary_loss_type=teacher_config.get('summary_loss_type', 'CE'),
            )
            self.distillation_loss_fns.append(loss_fn)
            logger.info(f"Created distillation loss for teacher {idx}: "
                       f"type={teacher_config['loss_type']}, mode={teacher_config['mode']}")
        
        # For backward compatibility, keep single loss reference if only one teacher
        if len(self.distillation_loss_fns) == 1:
            self.distillation_loss_fn = self.distillation_loss_fns[0]

    @staticmethod
    def _get_parameter_groups(model, weight_decay, skip_names=()):
        decay = []
        no_decay = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if any(s in name for s in skip_names):
                no_decay.append(param)
            else:
                decay.append(param)

        return [
            {"params": no_decay, "weight_decay": 0.0},
            {"params": decay, "weight_decay": weight_decay},
        ]

    def _merge_parameter_groups(self, *param_groups):
        """Merge multiple parameter group lists into a single list.

        Each input is a list of two dicts: [{"params": no_decay, ...}, {"params": decay, ...}].
        The no_decay params and decay params are concatenated separately.
        """
        merged_no_decay = []
        merged_decay = []
        weight_decay = 0.0
        for groups in param_groups:
            merged_no_decay.extend(groups[0]["params"])
            merged_decay.extend(groups[1]["params"])
            weight_decay = groups[1]["weight_decay"]
        return [
            {"params": merged_no_decay, "weight_decay": 0.0},
            {"params": merged_decay, "weight_decay": weight_decay},
        ]

    def configure_optimizers(self):
        """Configure optimizers for training"""
        student_params = self._get_parameter_groups(
            self.model, self.optimizer.weight_decay, self.optimizer.skip_names
        )
        # Include distillation loss parameters (projection MLPs, etc.)
        distill_params = self._get_parameter_groups(
            self.distillation_loss_fns, self.optimizer.weight_decay, self.optimizer.skip_names
        )
        parameters = self._merge_parameter_groups(student_params, distill_params)
        # define optimizers
        if self.optimizer.optim == "sgd":
            self.optimizer_G = optim.SGD(
                parameters,
                lr=self.lr,
                momentum=self.optimizer.momentum,  # 0.9
                weight_decay=self.optimizer.weight_decay,
            )  # 5e-4
        elif self.optimizer.optim == "adam":
            self.optimizer_G = optim.Adam(
                parameters,
                lr=self.lr,
                weight_decay=self.optimizer.weight_decay,
            )  # 0
        elif self.optimizer.optim == "adamw":
            self.optimizer_G = optim.AdamW(
                parameters,
                lr=self.lr,
                betas=self.optimizer.betas,
                weight_decay=self.optimizer.weight_decay,
            )
        else:
            raise NotImplementedError(
                "Optimizer {} is not implemented".format(self.optimizer.optim)
            )

        # Create main scheduler based on policy
        lr_policy = self.lr_policy.lower()
        if lr_policy == "linear":

            def lambda_rule(epoch):
                # gradually decay learning rate from epoch 0 to max_epochs
                lr_l = 1 - (epoch) / float(self.max_epochs + 1)
                return lr_l

            scheduler = lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lambda_rule)
        elif lr_policy == "step":
            interval = "epoch"
            if self.lr_policy_params is not None:
                step_size = self.lr_policy_params.step_size
                gamma = self.lr_policy_params.gamma
            else:   # default values
                step_size = self.max_epochs // 4
                gamma = 0.1
            # args.lr_decay_iters
            scheduler = lr_scheduler.StepLR(
                self.optimizer_G, step_size=step_size, gamma=gamma
            )
        elif lr_policy == "multistep":
            interval = "epoch"
            if self.lr_policy_params is not None:
                milestones = self.lr_policy_params.milestones
                gamma = self.lr_policy_params.gamma
            else:
                milestones = [self.max_epochs // 2]
                gamma = 0.1
            scheduler = lr_scheduler.MultiStepLR(self.optimizer_G, milestones, gamma=gamma)
        elif lr_policy == "cosine":
            interval = "step"
            epoch_steps = self.trainer.estimated_stepping_batches // (self.trainer.max_epochs * self.trainer.accumulate_grad_batches)
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer_G,
                num_training_steps=self.trainer.estimated_stepping_batches,
                num_warmup_steps=epoch_steps * self.optimizer.warmup_epochs,
            )
        else:
            raise NotImplementedError('learning rate policy [{}] is not implemented'.format(self.lr_policy))

        self.lr_scheduler = scheduler

        optim_dict = {}
        optim_dict["optimizer"] = self.optimizer_G
        optim_dict["lr_scheduler"] = {
            "scheduler": self.lr_scheduler,
            "interval": interval,
            "frequency": 1
        }
        optim_dict["monitor"] = self.monitor_name
        return optim_dict

    def _get_stochastic_resolution(self):
        """Return (res_list, prob_list) for per-batch resize, or (None, None) to skip.
        When augmentation.stochastic_resolutions is set (global), or any teacher has
        stochastic_resolutions (per-teacher / multiview), the dataloader does per-sample
        resolution sampling, so we skip per-batch resize here. Otherwise use multi_scales if set.
        """
        aug = self.experiment_spec.dataset.augmentation
        stoch = getattr(aug, "stochastic_resolutions", None)
        if stoch is not None and hasattr(stoch, 'items') and len(stoch) > 0:
            return None, None
        # Per-teacher stochastic_resolutions: multiview pipeline already does per-sample resolution
        if any(c.get("stochastic_resolutions") for c in self.teacher_configs):
            return None, None
        multi_scales = list(getattr(aug, "multi_scales", []) or [])
        if not multi_scales:
            return None, None
        res = [list(i.keys())[0] for i in multi_scales]
        prob = [list(i.values())[0] for i in multi_scales]
        total = sum(prob)
        if total <= 0:
            return None, None
        prob = [p / total for p in prob]
        return res, prob

    @staticmethod
    def _dump_batch(dump_dir, batch_idx, batch):
        """Save a batch to disk for parity testing."""
        os.makedirs(dump_dir, exist_ok=True)
        data = {
            "student_img": batch["img"].detach().cpu(),
            "class": batch.get("class", torch.tensor([])).detach().cpu(),
        }
        if "valid_mask" in batch:
            data["student_mask"] = batch["valid_mask"].detach().cpu()
        tvs = batch.get("teacher_views", [])
        data["num_teachers"] = len(tvs)
        for i, tv in enumerate(tvs):
            data[f"teacher_{i}_img"] = tv["img"].detach().cpu()
            data[f"teacher_{i}_mask"] = tv["valid_mask"].detach().cpu()
            data[f"teacher_{i}_stm"] = tv["spatial_transform"].detach().cpu()
        path = os.path.join(dump_dir, f"batch_{batch_idx:03d}.pt")
        torch.save(data, path)
        if batch_idx == 0:
            logger = logging.getLogger("parity_batch_dump")
            logger.info("Parity batch dump: saving to %s", dump_dir)

    def training_step(self, batch, batch_idx):
        """Training step"""
        _dump_dir = os.environ.get("PARITY_DUMP_DIR")
        if _dump_dir and batch_idx < int(os.environ.get("PARITY_DUMP_BATCHES", "5")):
            self._dump_batch(_dump_dir, batch_idx, batch)

        res_list, prob_list = self._get_stochastic_resolution()
        if res_list is not None and prob_list is not None:
            sz = int(np.random.choice(a=res_list, p=prob_list))
            if isinstance(sz, int):
                batch["img"] = F.interpolate(batch["img"], size=[sz, sz])
            elif isinstance(sz, (list, tuple)):
                batch["img"] = F.interpolate(batch["img"], size=sz)
            else:
                raise TypeError(f"{sz} is {type(sz)}. Need to pass int / list / tuple for multi_scale")

        student_input = self._normalize_student_input(batch["img"])
        student_summary, student_spatial = self.model(student_input, return_features=True)
        if self.num_classes > 0:
            out = self.model.head(student_summary)
            acc = self.train_acc(out, batch["class"].long())
            self.log_dict(acc, sync_dist=False, on_step=True, on_epoch=False, prog_bar=True)
            loss = self.criterion(out, batch["class"].long())
        else:
            loss = torch.tensor(0.0).to(student_summary.device)
        
        # Compute distillation loss from all teachers
        total_distillation_loss = torch.tensor(0.0, device=loss.device if torch.is_tensor(loss) else 'cpu')
        total_teacher_weight = 0.0
        distill_scale = 1.0
        use_multiview = "teacher_views" in batch and len(batch["teacher_views"]) > 0

        for idx, (loss_fn, teacher_config) in enumerate(zip(self.distillation_loss_fns, self.teacher_configs)):
            if use_multiview and idx < len(batch["teacher_views"]):
                tv = batch["teacher_views"][idx]
                teacher_input = tv["img"].to(batch["img"].device)
                teacher_input = self._apply_teacher_normalization(
                    teacher_input, teacher_config, batch["img"].device
                )
                if "valid_mask" in batch:
                    student_valid_mask = batch["valid_mask"].to(batch["img"].device)
                    if student_valid_mask.dim() == 4 and student_valid_mask.shape[1] == 1:
                        student_valid_mask = student_valid_mask.squeeze(1)
                else:
                    student_valid_mask = torch.ones(
                        batch["img"].shape[0], batch["img"].shape[2], batch["img"].shape[3],
                        dtype=torch.float32, device=batch["img"].device
                    )
                teacher_valid_mask = tv["valid_mask"].to(batch["img"].device)
                if teacher_valid_mask.dim() == 4 and teacher_valid_mask.shape[1] == 1:
                    teacher_valid_mask = teacher_valid_mask.squeeze(1)
                spatial_transform = tv["spatial_transform"].to(batch["img"].device)
                teacher_distill_loss = loss_fn(
                    batch["img"],
                    teacher_batch_input=teacher_input,
                    student_valid_mask=student_valid_mask,
                    teacher_valid_mask=teacher_valid_mask,
                    spatial_transform=spatial_transform,
                    student_summary=student_summary,
                    student_spatial=student_spatial,
                )
            else:
                teacher_input = self._apply_teacher_normalization(
                    batch["img"], teacher_config, batch["img"].device
                )
                teacher_distill_loss = loss_fn(
                    batch["img"],
                    teacher_batch_input=teacher_input,
                    student_summary=student_summary,
                    student_spatial=student_spatial,
                )
            teacher_weight = teacher_config['loss_lambda']
            weighted_loss = teacher_weight * teacher_distill_loss * distill_scale

            if not torch.isnan(weighted_loss):
                total_distillation_loss += weighted_loss
                total_teacher_weight += teacher_weight

            # Log per-teacher loss (unscaled, for readability)
            self.log(
                f"distill_loss_teacher_{idx}",
                teacher_distill_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=self.batch_size,
                rank_zero_only=True
            )
        
        # Normalize supervised loss weight
        if total_teacher_weight > 0:
            supervised_weight = 1.0 - total_teacher_weight
        else:
            supervised_weight = 1.0
        
        supervised_loss = supervised_weight * loss
        distill_loss = total_distillation_loss
        # Unscaled distillation loss (raw value from loss_fn, ~1–3) for logging/prog_bar
        distill_loss_raw = distill_loss / distill_scale if total_teacher_weight > 0 else distill_loss

        if torch.isnan(supervised_loss):
            supervised_loss = torch.tensor(0.0)

        if torch.isnan(distill_loss):
            distill_loss = torch.tensor(0.0)

        total_loss = supervised_loss + distill_loss
        self.log(
            "distillation_loss_raw",
            distill_loss_raw,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.batch_size,
            rank_zero_only=True,
        )
        self.log(
            "lr",
            self.lr_schedulers().get_last_lr()[-1],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True
        )
        self.log(
            "supervised_loss",
            supervised_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.batch_size,
            rank_zero_only=True
        )
        self.log(
            "distillation_loss",
            distill_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.batch_size,
            rank_zero_only=True
        )
        self.log(
            "total_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.batch_size,
            rank_zero_only=True
        )
        return {"loss": total_loss}

    def on_train_epoch_start(self):
        """Log epoch counter for milestone checkpoint pruning (keep last N)."""
        self.log(
            "epoch_counter",
            float(self.current_epoch),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
            rank_zero_only=True,
        )

    def on_train_epoch_end(self):
        """Log Training metrics to status.json"""
        average_train_loss = self.trainer.logged_metrics["total_loss_epoch"].item()
        self.train_acc.reset()
        self.status_logging_dict = {}
        self.status_logging_dict["train_loss"] = average_train_loss

        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Train metrics generated.",
            status_level=status_logging.Status.RUNNING,
        )

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        student_input = self._normalize_student_input(batch["img"])
        student_summary, student_spatial = self.model(student_input, return_features=True)
        if self.num_classes > 0:
            out = self.model.head(student_summary)
            loss = self.criterion(out, batch["class"].long())
            self.valid_acc.update(out, batch["class"].long())
            self.log(
                "val_loss",
                loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=True,
                batch_size=self.batch_size,
                rank_zero_only=True
            )
        else:
            out = student_summary
            loss = torch.tensor(0.0).to(out.device)

        # Compute distillation loss from all teachers
        total_distillation_loss = torch.tensor(0.0, device=out.device)
        use_multiview = "teacher_views" in batch and len(batch["teacher_views"]) > 0

        for idx, (loss_fn, teacher_config) in enumerate(zip(self.distillation_loss_fns, self.teacher_configs)):
            if use_multiview and idx < len(batch["teacher_views"]):
                tv = batch["teacher_views"][idx]
                teacher_input = tv["img"].to(out.device)
                teacher_input = self._apply_teacher_normalization(
                    teacher_input, teacher_config, out.device
                )
                student_valid_mask = torch.ones(
                    batch["img"].shape[0], batch["img"].shape[2], batch["img"].shape[3],
                    dtype=torch.float32, device=out.device
                )
                teacher_valid_mask = tv["valid_mask"].to(out.device)
                if teacher_valid_mask.dim() == 3:
                    teacher_valid_mask = teacher_valid_mask[:, 0]
                spatial_transform = tv["spatial_transform"].to(out.device)
                teacher_distill_loss = loss_fn(
                    batch["img"],
                    teacher_batch_input=teacher_input,
                    student_valid_mask=student_valid_mask,
                    teacher_valid_mask=teacher_valid_mask,
                    spatial_transform=spatial_transform,
                    student_summary=student_summary,
                    student_spatial=student_spatial,
                )
            else:
                teacher_input = self._apply_teacher_normalization(
                    batch["img"], teacher_config, out.device
                )
                teacher_distill_loss = loss_fn(
                    batch["img"],
                    teacher_batch_input=teacher_input,
                    student_summary=student_summary,
                    student_spatial=student_spatial,
                )

            if not torch.isnan(teacher_distill_loss):
                total_distillation_loss += teacher_distill_loss
            
            # Log per-teacher validation loss
            self.log(
                f"val_distill_loss_teacher_{idx}",
                teacher_distill_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
                batch_size=self.batch_size,
                rank_zero_only=True
            )
        
        # Log total distillation loss
        self.log(
            "distillation_loss",
            total_distillation_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.batch_size,
            rank_zero_only=True
        )
        loss += total_distillation_loss
        return loss

    def on_validation_epoch_end(self):
        """Validation epoch end.
        compute mAP at the end of epoch
        """
        # FLUSHING VALIDATION EPOCH METRICS
        # scores, mean_scores = self._collect_epoch_states()  # logs all evaluation metrics
        # self.log("val_acc", scores['acc'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.num_classes > 0:
            acc = self.valid_acc.compute()
            self.log_dict(acc, sync_dist=True, on_step=False, on_epoch=True, prog_bar=False)
            self.valid_acc.reset()
            # self._clear_cache()

            average_val_loss = self.trainer.logged_metrics["val_loss"].item()
            if not self.trainer.sanity_checking:
                self.status_logging_dict = {}
                self.status_logging_dict["val_loss"] = average_val_loss
                for acc_key in acc.keys():
                    self.status_logging_dict[acc_key] = acc[acc_key].item()
                status_logging.get_status_logger().kpi = self.status_logging_dict
                status_logging.get_status_logger().write(
                    message="Eval metrics generated.",
                    status_level=status_logging.Status.RUNNING,
                )

        pl.utilities.memory.garbage_collection_cuda()

    def on_save_checkpoint(self, checkpoint):
        """Save the checkpoint but ignore the teacher weights."""
        keys_to_pop = [
            key for key in checkpoint["state_dict"].keys() 
            if key.startswith("teacher") or key.startswith("teachers")
        ]
        for key in keys_to_pop:
            checkpoint["state_dict"].pop(key)
        checkpoint["tao_model"] = "classification"
