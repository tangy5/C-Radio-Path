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

"""Normalization presets for different domains and foundation models.

This module provides standard normalization values used by various
foundation models for pathology and natural image tasks.
"""

from typing import List, Tuple, Dict, Any

# =============================================================================
# Pathology H&E Stain Normalization
# =============================================================================

# H-optimus-0 (Bioptimus) - ViT-Giant trained on 500k+ pathology WSIs
# Optimized for Hematoxylin & Eosin (H&E) stained slides
PATHOLOGY_H_OPTIMUS_MEAN: List[float] = [0.707223, 0.578729, 0.703617]
PATHOLOGY_H_OPTIMUS_STD: List[float] = [0.211883, 0.230117, 0.177517]

# Prov-GigaPath - Uses ImageNet normalization
PATHOLOGY_GIGAPATH_MEAN: List[float] = [0.485, 0.456, 0.406]
PATHOLOGY_GIGAPATH_STD: List[float] = [0.229, 0.224, 0.225]

# UNI2-h - Uses ImageNet normalization  
PATHOLOGY_UNI2H_MEAN: List[float] = [0.485, 0.456, 0.406]
PATHOLOGY_UNI2H_STD: List[float] = [0.229, 0.224, 0.225]


# =============================================================================
# Natural Image Normalization
# =============================================================================

# Standard ImageNet (ResNet, etc.)
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

# OpenAI CLIP (SigLIP, SigLIP2, DINOv3)
OPENAI_CLIP_MEAN: List[float] = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD: List[float] = [0.26862954, 0.26130258, 0.27577711]

# =============================================================================
# Model-Specific Normalization
# =============================================================================

# C-RADIO v2/v3 default
C_RADIO_V2_MEAN: List[float] = [0.5, 0.5, 0.5]
C_RADIO_V2_STD: List[float] = [0.5, 0.5, 0.5]

# SigLIP2 normalization (matches CLIP)
SIGLIP2_MEAN: List[float] = OPENAI_CLIP_MEAN
SIGLIP2_STD: List[float] = OPENAI_CLIP_STD

# DINOv3 normalization (matches CLIP)
DINOV3_MEAN: List[float] = OPENAI_CLIP_MEAN
DINOV3_STD: List[float] = OPENAI_CLIP_STD

# RADIO family (varies by version)
RADIO_V2_MEAN: List[float] = [0.5, 0.5, 0.5]
RADIO_V2_STD: List[float] = [0.5, 0.5, 0.5]

RADIO_V3_MEAN: List[float] = OPENAI_CLIP_MEAN
RADIO_V3_STD: List[float] = OPENAI_CLIP_STD

# =============================================================================
# Preset Registry
# =============================================================================

NORMALIZATION_PRESETS: Dict[str, Dict[str, Any]] = {
    # Pathology presets
    "pathology_h_optimus": {
        "mean": PATHOLOGY_H_OPTIMUS_MEAN,
        "std": PATHOLOGY_H_OPTIMUS_STD,
        "description": "H&E pathology optimized (H-optimus-0)",
        "source": "Bioptimus H-optimus-0",
    },
    "pathology_gigapath": {
        "mean": PATHOLOGY_GIGAPATH_MEAN,
        "std": PATHOLOGY_GIGAPATH_STD,
        "description": "Prov-GigaPath (ImageNet)",
        "source": "Microsoft/Providence",
    },
    "pathology_uni2h": {
        "mean": PATHOLOGY_UNI2H_MEAN,
        "std": PATHOLOGY_UNI2H_STD,
        "description": "UNI2-h (ImageNet)",
        "source": "Mahmood Lab/Harvard",
    },
    # Natural image presets
    "imagenet": {
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "description": "Standard ImageNet",
        "source": "ImageNet dataset",
    },
    "openai_clip": {
        "mean": OPENAI_CLIP_MEAN,
        "std": OPENAI_CLIP_STD,
        "description": "OpenAI CLIP",
        "source": "OpenAI CLIP",
    },
    # Model-specific presets
    "c_radio_v2": {
        "mean": C_RADIO_V2_MEAN,
        "std": C_RADIO_V2_STD,
        "description": "C-RADIO v2/v3 default",
        "source": "NVIDIA C-RADIO",
    },
    "siglip2": {
        "mean": SIGLIP2_MEAN,
        "std": SIGLIP2_STD,
        "description": "SigLIP2 (OpenAI CLIP)",
        "source": "Google SigLIP2",
    },
    "dinov3": {
        "mean": DINOV3_MEAN,
        "std": DINOV3_STD,
        "description": "DINOv3 (OpenAI CLIP)",
        "source": "Meta DINOv3",
    },
}


def get_normalization_preset(name: str) -> Tuple[List[float], List[float]]:
    """Get normalization preset by name.
    
    Args:
        name: Name of the preset (e.g., "pathology_h_optimus", "imagenet")
        
    Returns:
        Tuple of (mean, std) lists
        
    Raises:
        ValueError: If preset name is not recognized
        
    Example:
        >>> mean, std = get_normalization_preset("pathology_h_optimus")
        >>> print(mean)  # [0.707223, 0.578729, 0.703617]
    """
    if name not in NORMALIZATION_PRESETS:
        available = ", ".join(sorted(NORMALIZATION_PRESETS.keys()))
        raise ValueError(
            f"Unknown preset: '{name}'.\n"
            f"Available presets: {available}"
        )
    
    preset = NORMALIZATION_PRESETS[name]
    return preset["mean"].copy(), preset["std"].copy()


def list_normalization_presets() -> Dict[str, str]:
    """List all available normalization presets.
    
    Returns:
        Dictionary mapping preset names to their descriptions
        
    Example:
        >>> presets = list_normalization_presets()
        >>> for name, desc in presets.items():
        ...     print(f"{name}: {desc}")
    """
    return {
        name: f"{info['description']} (from {info['source']})"
        for name, info in sorted(NORMALIZATION_PRESETS.items())
    }


def get_teacher_normalization(teacher_type: str) -> Tuple[List[float], List[float]]:
    """Get recommended normalization for a specific teacher model.
    
    Args:
        teacher_type: Teacher model type (e.g., "h_optimus_0", "prov_gigapath")
        
    Returns:
        Tuple of (mean, std) lists
    """
    teacher_preset_map = {
        "h_optimus_0": "pathology_h_optimus",
        "prov_gigapath": "pathology_gigapath",
        "uni2_h": "pathology_uni2h",
        "c_radio_v2": "c_radio_v2",
        "c_radio_v3": "openai_clip",
        "c_radio_v4": "openai_clip",
        "siglip2": "siglip2",
        "dinov3": "dinov3",
    }
    
    preset_name = teacher_preset_map.get(teacher_type.lower(), "imagenet")
    return get_normalization_preset(preset_name)


# =============================================================================
# Backward Compatibility
# =============================================================================

# Keep backward compatibility with existing code
HOPTIMUS_MEAN = PATHOLOGY_H_OPTIMUS_MEAN
HOPTIMUS_STD = PATHOLOGY_H_OPTIMUS_STD


if __name__ == "__main__":
    # Print all available presets
    print("Available Normalization Presets:")
    print("=" * 60)
    for name, desc in list_normalization_presets().items():
        print(f"\n{name}:")
        print(f"  Description: {desc}")
        mean, std = get_normalization_preset(name)
        print(f"  Mean: {mean}")
        print(f"  Std:  {std}")
