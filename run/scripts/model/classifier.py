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

"""Classification model builder"""

import torch

from nvidia_tao_pytorch.core.distributed.comm import get_global_rank
from nvidia_tao_pytorch.core.tlt_logging import logger
from nvidia_tao_pytorch.core.utils.pos_embed_interpolation import interpolate_patch_embed, interpolate_pos_embed
from nvidia_tao_pytorch.core.utils.ptm_utils import load_pretrained_weights
from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.dino_v2 import DINOV2
from nvidia_tao_pytorch.cv.backbone_v2.radio import RADIO
from nvidia_tao_pytorch.cv.classification_pyt.model.utils import cls_parser, ptm_adapter


def _apply_radio_metadata(model, state_dict):
    """Apply non-parameter metadata from checkpoint to RADIO models.

    RADIO checkpoints may contain metadata like `summary_idxs` and
    `input_conditioner` normalization values that are not nn.Parameters
    and thus not loaded by load_state_dict(). This function extracts
    them from the state_dict and applies them to the model.

    Args:
        model: The model (possibly RADIO) to update.
        state_dict: The checkpoint state dict to read metadata from.
    """
    if not isinstance(model, RADIO):
        return

    # Try to find summary_idxs in checkpoint.
    # Keys may be prefixed with 'radio_model.' or 'radio.' depending on source.
    for prefix in ('radio_model.', 'radio.radio.', ''):
        key = f"{prefix}summary_idxs"
        if key in state_dict:
            ckpt_summary_idxs = state_dict[key].tolist()
            if ckpt_summary_idxs != model.summary_idxs:
                if get_global_rank() == 0:
                    logger.info(
                        f"Overriding summary_idxs from checkpoint: "
                        f"{model.summary_idxs} -> {ckpt_summary_idxs}"
                    )
                model.summary_idxs = ckpt_summary_idxs
                # Also update the inner RADIOBase module
                if hasattr(model, 'radio') and hasattr(model.radio, 'summary_idxs'):
                    model.radio.summary_idxs = ckpt_summary_idxs
                if hasattr(model, 'radio') and hasattr(model.radio, 'radio') and hasattr(model.radio.radio, 'summary_idxs'):
                    model.radio.radio.summary_idxs = ckpt_summary_idxs
                # Recompute num_features based on corrected summary_idxs
                old_num_features = model.num_features
                model.num_features = len(ckpt_summary_idxs) * model.radio.radio.model.embed_dim
                if get_global_rank() == 0:
                    logger.info(
                        f"Updated num_features: {old_num_features} -> {model.num_features} "
                        f"({len(ckpt_summary_idxs)} summary tokens x {model.radio.radio.model.embed_dim} dim)"
                    )
            break


def build_model(experiment_config,
                export=False):
    """ Build Classifier model according to configuration

    Args:
        experiment_config: experiment configuration
        export: flag to indicate onnx export

    Returns:
        model

    """
    model_config = experiment_config.model
    dataset_config = experiment_config.dataset

    backbone = model_config.backbone['type']
    freeze_backbone = model_config.backbone['freeze_backbone']
    freeze_norm = model_config.backbone['freeze_norm']
    pretrained_backbone_path = model_config.backbone.pretrained_backbone_path
    hf_model_path = model_config.backbone.get('hf_model_path', None)

    try:
        backbone_kwargs = dict(
            num_classes=dataset_config.num_classes,
            freeze_at='all' if freeze_backbone else None,
            freeze_norm=freeze_norm,
            export=export,
        )
        if hf_model_path is not None:
            backbone_kwargs['hf_model_path'] = hf_model_path
        model = BACKBONE_REGISTRY.get(backbone)(**backbone_kwargs)
        # We should unfreeze the head for training, even `freeze_backbone` is set to `True`.
        if freeze_backbone:
            head = model.get_classifier()
            for p in head.parameters():
                p.requires_grad = True
            head.train()
    except KeyError as e:
        logger.error(f"Error building model: {e}")
        logger.warning(f"BACKBONE_REGISTRY: {BACKBONE_REGISTRY}")
        raise e

    # Load pretrained backbone
    if pretrained_backbone_path:
        logger.info(f"Loading pretrained weights from {pretrained_backbone_path}")
        state_dict = load_pretrained_weights(pretrained_backbone_path, parser=cls_parser, ptm_adapter=ptm_adapter)
        if isinstance(model, DINOV2):
            state_dict = interpolate_vit_checkpoint(
                checkpoint=state_dict,
                target_patch_size=14,
                target_resolution=518,
            )
        msg = model.load_state_dict(state_dict, strict=False)
        # Apply non-parameter metadata from checkpoint (e.g. summary_idxs)
        _apply_radio_metadata(model, state_dict)
        if get_global_rank() == 0:
            logger.info(f"Loaded pretrained weights from {pretrained_backbone_path}")
            logger.info(f"{msg}")

    return model


def interpolate_vit_checkpoint(checkpoint, target_patch_size, target_resolution):
    """ Interpolate ViT backbone position embedding and patch embedding

    Args:
        checkpoint: pretrained ViT checkpoint
        target_patch_size: target patch size to interpolate to. ex: 14, 16, etc
        target_resolution: target image size to interpolate to. ex: 224, 512, 518, etc

    Returns:
        interpolated model checkpoints

    """
    if checkpoint is None:
        return checkpoint

    if get_global_rank() == 0:
        logger.info("Do ViT pretrained backbone interpolation")
    checkpoint = interpolate_patch_embed(checkpoint=checkpoint, new_patch_size=target_patch_size)
    checkpoint = interpolate_pos_embed(
        checkpoint_model=checkpoint, new_resolution=target_resolution, new_patch_size=target_patch_size
    )
    return checkpoint
