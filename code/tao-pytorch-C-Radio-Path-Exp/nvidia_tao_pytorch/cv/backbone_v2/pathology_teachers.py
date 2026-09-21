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

"""Pathology Foundation Model Backbones for Teacher Distillation

This module provides backbone implementations for pathology-specific
foundation models to be used as teachers in knowledge distillation:

1. H-optimus-0 (Bioptimus) - ViT-Giant with pathology-specific normalization
2. Prov-GigaPath (Microsoft/Providence) - ViT-Giant DINOv2
3. UNI2-h (Mahmood Lab / Harvard) - ViT-Giant DINOv2 

All models are loaded via timm and wrapped to provide a consistent interface
for the TAO distillation pipeline.
"""

import os
from typing import Tuple, Optional, List, Dict

import torch
from torch import nn
import timm
from timm.layers import SwiGLUPacked

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase


class PathologyModelWrapper(BackboneBase):
    """Generic wrapper for pathology foundation models.
    
    Provides consistent interface for timm-based pathology models
    with support for feature extraction and proper normalization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_features: int,
        patch_size: int = 14,
        num_classes: int = 0,
        in_chans: int = 3,
        activation_checkpoint: bool = False,
        freeze_at: Optional[list] = None,
        freeze_norm: bool = False,
        norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        norm_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        **kwargs
    ):
        super().__init__(
            in_chans=in_chans,
            num_classes=num_classes,
            activation_checkpoint=activation_checkpoint,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
        )
        self.inner = model
        self._num_features = num_features
        self._patch_size = patch_size
        self._norm_mean = norm_mean
        self._norm_std = norm_std
        
        # Register normalization buffers for easy access
        self.register_buffer('norm_mean_buf', torch.tensor(norm_mean).view(1, 3, 1, 1))
        self.register_buffer('norm_std_buf', torch.tensor(norm_std).view(1, 3, 1, 1))
        
        # Head is identity for feature extraction (teacher models)
        self.head = nn.Identity()
        
    @property
    def num_features(self) -> int:
        """Return the feature dimension of the model."""
        return self._num_features
    
    @property
    def patch_size(self) -> int:
        """Return the patch size of the ViT."""
        return self._patch_size
    
    @property
    def norm_mean(self) -> Tuple[float, float, float]:
        """Return normalization mean values."""
        return self._norm_mean
    
    @property
    def norm_std(self) -> Tuple[float, float, float]:
        """Return normalization std values."""
        return self._norm_std
    
    def get_stage_dict(self):
        """Get the stage dictionary for feature extraction."""
        # For ViT models, return the transformer blocks
        if hasattr(self.inner, 'blocks'):
            stage_dict = {0: self.inner.patch_embed}
            for i, block in enumerate(self.inner.blocks, start=1):
                stage_dict[i] = block
            if hasattr(self.inner, 'norm'):
                stage_dict[len(self.inner.blocks) + 1] = self.inner.norm
            return stage_dict
        return {0: self.inner}
    
    def _get_num_prefix_tokens(self):
        """Get the number of prefix tokens (CLS + register tokens) from the inner model.

        In timm 1.0.x, the token layout after forward_features is:
        [CLS, reg1, ..., regN, patch1, ..., patchK]
        The 'num_prefix_tokens' property gives 1 (CLS) + N (register tokens).
        """
        return getattr(self.inner, 'num_prefix_tokens', 1)

    def _extract_spatial_features(self, features, img_h, img_w):
        """Extract spatial patch tokens from the full feature sequence.

        Skips prefix tokens (CLS + registers) and reshapes to NCHW format.

        Args:
            features: Full token sequence [B, N_total, D] from forward_features.
            img_h: Input image height (used to compute patch grid).
            img_w: Input image width (used to compute patch grid).

        Returns:
            Spatial feature map [B, D, H_patches, W_patches].
        """
        B, N, D = features.shape
        num_prefix = self._get_num_prefix_tokens()
        H_patches = img_h // self._patch_size
        W_patches = img_w // self._patch_size
        num_spatial = H_patches * W_patches

        spatial = features[:, num_prefix:num_prefix + num_spatial, :]  # [B, H*W, D]
        spatial = spatial.permute(0, 2, 1).view(B, D, H_patches, W_patches)  # [B, C, H, W]
        return spatial

    def forward_feature_pyramid(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning feature pyramid.

        For ViT models, returns the spatial features reshaped to [B, C, H, W] format
        to be compatible with TAO's spatial distillation mode.
        """
        features = self.forward_features(x)  # [B, N, D]
        spatial_features = self._extract_spatial_features(features, x.shape[2], x.shape[3])
        return {"stage_final": spatial_features}
    
    @torch.jit.ignore
    def get_classifier(self):
        """Get the classification head."""
        return self.head
    
    def reset_classifier(self, num_classes: int, global_pool: str = ""):
        """Reset the classification head."""
        self.num_classes = num_classes
        self.head = nn.Linear(self._num_features, num_classes) if num_classes > 0 else nn.Identity()
    
    def forward_pre_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone, excluding head.
        
        Returns:
            Tensor: Summary features (CLS token)
        """
        # Get CLS token output
        features = self.inner.forward_features(x)
        if isinstance(features, tuple):
            features = features[0]
        # Extract CLS token (first token)
        cls_token = features[:, 0]
        return cls_token
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning all features including spatial tokens.
        
        Returns:
            Tensor: All tokens [B, N+1, D] where N is number of patches
        """
        features = self.inner.forward_features(x)
        if isinstance(features, tuple):
            features = features[0]
        return features
    
    def forward(self, x: torch.Tensor, return_features: bool = False, return_logits: bool = False):
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W]
            return_features: If True, return (summary, spatial_features) tuple
            return_logits: If True, return pre-logits (CLS token)

        Returns:
            Tensor or tuple of tensors depending on flags.
            When return_features=True, returns (summary, spatial_features) where
            summary is [B, D] CLS token and spatial_features is [B, C, H, W].
        """
        if return_features:
            # Get all features and extract both summary and spatial features
            features = self.forward_features(x)  # [B, N, D]

            # Summary: CLS token (first token)
            summary = features[:, 0]  # [B, D]

            # Spatial features: patch tokens (skip CLS + register prefix tokens)
            spatial = self._extract_spatial_features(features, x.shape[2], x.shape[3])

            return summary, spatial

        summary = self.forward_pre_logits(x)

        if return_logits:
            return summary

        return self.head(summary)

    def load_state_dict(self, state_dict, strict=True):
        """Override to handle flat checkpoint keys that need 'inner.' prefix.

        Pathology foundation model checkpoints (H-optimus, GigaPath, UNI2)
        have flat keys (e.g. 'cls_token', 'blocks.0.attn.qkv.weight') but the wrapper
        stores the inner model as self.inner, so all keys are prefixed with 'inner.'.
        This method remaps flat keys to the expected 'inner.' prefixed format.
        """
        model_keys = set(super().state_dict().keys())

        new_state_dict = {}
        remapped = False
        for k, v in state_dict.items():
            if k not in model_keys and f'inner.{k}' in model_keys:
                new_state_dict[f'inner.{k}'] = v
                remapped = True
            else:
                new_state_dict[k] = v

        if remapped:
            print(f"  PathologyModelWrapper.load_state_dict: Remapped {sum(1 for k in state_dict if k not in model_keys and f'inner.{k}' in model_keys)} keys with 'inner.' prefix")

        return super().load_state_dict(new_state_dict, strict=strict)


def _load_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    """Load checkpoint into model, handling various checkpoint formats.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
    """
    print(f"  Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state dict from various checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DDP training)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load with strict=False to handle minor mismatches
    incompatible = model.load_state_dict(state_dict, strict=False)
    
    if incompatible.missing_keys:
        print(f"  Warning: Missing keys: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"  Warning: Unexpected keys: {len(incompatible.unexpected_keys)}")


# =============================================================================
# H-optimus-0 (Bioptimus)
# =============================================================================

@BACKBONE_REGISTRY.register()
def h_optimus_0(
    pretrained_backbone_path: Optional[str] = None,
    hf_model_path: Optional[str] = None,
    num_classes: int = 0,
    **kwargs
) -> PathologyModelWrapper:
    """H-optimus-0 pathology foundation model from Bioptimus.
    
    Architecture: ViT-Giant (DINOv2) with register tokens
    - Params: ~1.1B
    - Embed dim: 1536
    - Depth: 40 layers
    - Heads: 24
    - Patch size: 14
    - Input size: 224x224
    - Special: Pathology-specific normalization (H&E optimized)
    
    Args:
        pretrained_backbone_path: Local path to pytorch_model.bin (loaded separately by TAO)
        hf_model_path: HuggingFace Hub path (e.g., 'hf-hub:bioptimus/H-optimus-0')
        num_classes: Number of output classes (0 for feature extraction)
        **kwargs: Additional arguments
        
    Returns:
        PathologyModelWrapper wrapping H-optimus-0
    """
    # H-optimus-0 specific normalization for H&E pathology images
    hoptimus_mean = (0.707223, 0.578729, 0.703617)
    hoptimus_std = (0.211883, 0.230117, 0.177517)
    
    checkpoint_path = pretrained_backbone_path or hf_model_path
    
    # Create model architecture
    # TAO loads weights separately after model creation
    print(f"Creating H-optimus-0 model architecture...")
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        # Load with weights if checkpoint provided (direct call mode)
        print(f"  Loading weights from: {checkpoint_path}")
        model = timm.create_model(
            'vit_giant_patch14_reg4_dinov2',
            pretrained=False,
            checkpoint_path=checkpoint_path,
            num_classes=0,
            img_size=224,
        )
    else:
        # Create architecture only (TAO will load weights separately)
        print(f"  Creating architecture only (checkpoint will be loaded by TAO)")
        model = timm.create_model(
            'vit_giant_patch14_reg4_dinov2',
            pretrained=False,
            num_classes=0,
            img_size=224,
        )
    
    print(f"  Model created with pos_embed: {model.pos_embed.shape}")
    
    wrapper = PathologyModelWrapper(
        model=model,
        num_features=1536,
        patch_size=14,
        num_classes=num_classes,
        norm_mean=hoptimus_mean,
        norm_std=hoptimus_std,
        **kwargs
    )
    
    return wrapper


# =============================================================================
# Prov-GigaPath (Microsoft/Providence)
# =============================================================================

@BACKBONE_REGISTRY.register()
def prov_gigapath(
    pretrained_backbone_path: Optional[str] = None,
    hf_model_path: Optional[str] = None,
    num_classes: int = 0,
    **kwargs
) -> PathologyModelWrapper:
    """Prov-GigaPath pathology foundation model.
    
    Architecture: ViT-Giant (DINOv2)
    - Params: ~1.1B
    - Embed dim: 1536
    - Depth: 40 layers
    - Heads: 24
    - Patch size: 16 (despite name containing 'patch14')
    - Input size: 224x224
    - MLP ratio: 5.333
    
    Args:
        pretrained_backbone_path: Local path to pytorch_model.bin (loaded separately by TAO)
        hf_model_path: HuggingFace Hub path (e.g., 'hf-hub:prov-gigapath/prov-gigapath')
        num_classes: Number of output classes (0 for feature extraction)
        **kwargs: Additional arguments
        
    Returns:
        PathologyModelWrapper wrapping Prov-GigaPath
    """
    # Standard ImageNet normalization
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    
    checkpoint_path = pretrained_backbone_path or hf_model_path
    
    print(f"Creating Prov-GigaPath model architecture...")
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"  Loading weights from: {checkpoint_path}")
        model = timm.create_model(
            'vit_giant_patch14_dinov2',
            pretrained=False,
            checkpoint_path=checkpoint_path,
            num_classes=0,
            img_size=224,
            patch_size=16,
        )
    else:
        print(f"  Creating architecture only (checkpoint will be loaded by TAO)")
        model = timm.create_model(
            'vit_giant_patch14_dinov2',
            pretrained=False,
            num_classes=0,
            img_size=224,
            patch_size=16,
        )
    
    print(f"  Model created with pos_embed: {model.pos_embed.shape}")
    
    wrapper = PathologyModelWrapper(
        model=model,
        num_features=1536,
        patch_size=16,
        num_classes=num_classes,
        norm_mean=imagenet_mean,
        norm_std=imagenet_std,
        **kwargs
    )
    
    return wrapper


# =============================================================================
# UNI2-h (Mahmood Lab / Harvard)
# =============================================================================

@BACKBONE_REGISTRY.register()
def uni2_h(
    pretrained_backbone_path: Optional[str] = None,
    hf_model_path: Optional[str] = None,
    num_classes: int = 0,
    **kwargs
) -> PathologyModelWrapper:
    """UNI2-h pathology foundation model from Mahmood Lab.
    
    Architecture: ViT with 8 register tokens (DINOv2-based)
    - Params: ~600M
    - Embed dim: 1536
    - Depth: 24 layers
    - Heads: 16
    - Patch size: 14
    - Input size: 224x224
    - Special: 8 register tokens, GluMlp
    - Pretraining: 200M+ tiles from 350k slides (MGB)
    
    Args:
        pretrained_backbone_path: Local path to pytorch_model.bin (loaded separately by TAO)
        hf_model_path: HuggingFace Hub path (e.g., 'hf-hub:MahmoodLab/UNI2-h')
        num_classes: Number of output classes (0 for feature extraction)
        **kwargs: Additional arguments
        
    Returns:
        PathologyModelWrapper wrapping UNI2-h
    """
    # Standard ImageNet normalization
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    
    checkpoint_path = pretrained_backbone_path or hf_model_path
    
    print(f"Creating UNI2-h model architecture...")
    
    # UNI2-h has custom architecture: 24 layers, 8 registers, 1536 dim
    from timm.models.vision_transformer import VisionTransformer
    from timm.layers.mlp import GluMlp
    
    model = VisionTransformer(
        img_size=224,
        patch_size=14,
        embed_dim=1536,
        depth=24,
        num_heads=16,
        num_classes=0,
        mlp_ratio=16/3,  # = 5.333... to get hidden dim of 8192
        init_values=1e-5,
        reg_tokens=8,  # UNI2-h has 8 register tokens
        global_pool='token',
        dynamic_img_size=False,
        mlp_layer=GluMlp,  # DINOv2 uses GluMlp
        no_embed_class=True,  # Don't include CLS/registers in pos_embed
    )
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"  Loading weights from: {checkpoint_path}")
        _load_checkpoint(model, checkpoint_path)
    else:
        print(f"  Architecture created (checkpoint will be loaded by TAO)")
    
    print(f"  Model created with pos_embed: {model.pos_embed.shape}")
    
    wrapper = PathologyModelWrapper(
        model=model,
        num_features=1536,
        patch_size=14,
        num_classes=num_classes,
        norm_mean=imagenet_mean,
        norm_std=imagenet_std,
        **kwargs
    )
    
    return wrapper


