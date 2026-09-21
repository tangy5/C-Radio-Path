import inspect
import os
import numpy as np
import string
from typing import Dict, List, Union, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase


class SigLIP2Wrapper(BackboneBase):
    def __init__(self,
                 clip_model,
                 tokenizer,
                 num_classes: int = 0,
                 patch_size: int = 16,
                 is_dynamic: bool = True,
                 in_chans: int = 3,
                 activation_checkpoint=False,
                 freeze_at=None,
                 freeze_norm=False,
                 head_init_scale=1.0,
                 **kwargs,
                 ):
        super().__init__(
            in_chans=in_chans,
            num_classes=num_classes,
            activation_checkpoint=activation_checkpoint,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
        )
        self.inner = clip_model
        self.tokenizer = tokenizer

        self._patch_size = patch_size
        self._is_dynamic = is_dynamic
        self.num_features = self.inner.vision_model.config.hidden_size
        # print(f"num_features: {self.num_features}") # 1152
        self.register_buffer('mask', torch.ones(1, 1, dtype=torch.int32))
        if num_classes > 0:
            self.head = nn.Linear(self.num_features, num_classes)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)
        else:
            self.head = nn.Identity()

    @property
    def patch_size(self):
        return self._patch_size

    def get_stage_dict(self):
        """Get the stage dictionary."""
        stage_dict = {0: self.inner.vision_model.embeddings}
        for i, block in enumerate(self.inner.vision_model.encoder.layers, start=1):
            stage_dict[i] = block
        return stage_dict

    @torch.jit.ignore
    def get_classifier(self):
        """Get the classification head module.

        Returns:
            nn.Module: The classification head (Linear layer or Identity).
        """
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        """Reset the classification head with a new number of classes.

        Args:
            num_classes (int): New number of classes for classification.
            global_pool (str, optional): Global pooling type (unused in current implementation).
                Defaults to "".
        """
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_pre_logits(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the backbone, excluding the head.

        Args:
            x (Tensor): Input tensor.

        Returns:
            summary (Tensor): Summary tensor.
            features (Tensor): Features tensor.
        """
        summary = self.forward(x, return_features=False, return_logits=True)
        return summary

    def forward(self, x: torch.Tensor, return_features: bool = False, return_logits: bool = False):
        out_h = x.shape[-2] // self._patch_size
        out_w = x.shape[-1] // self._patch_size

        extra = dict()

        if self._is_dynamic:
            pixel_values = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                                     p1=self._patch_size, p2=self._patch_size,
                                     h=out_h, w=out_w)
            mask = self.mask.expand(*pixel_values.shape[:2])
            shapes = torch.tensor([(out_h, out_w)] * pixel_values.shape[0], dtype=torch.int64, device=x.device)

            extra = dict(attention_mask=mask, spatial_shapes=shapes)
        else:
            pixel_values = x
        sig = inspect.signature(self.inner.vision_model.forward)
        if 'return_dict' in sig.parameters:
            extra['return_dict'] = True
        output = self.inner.vision_model(pixel_values=pixel_values, **extra)

        summary = output.pooler_output
        if return_features:
            features = output.last_hidden_state
            # if kwargs.get('feature_fmt', None) == 'NCHW':
            features = rearrange(features, 'b (h w) c -> b c h w', h=out_h, w=out_w)
            return summary, features
        else:
            if return_logits:
                return summary
            else:
                return self.head(summary)

    def forward_feature_pyramid(self, x: torch.Tensor):
        _, features = self.forward(x, return_features=True, return_logits=False)
        return features

    @torch.jit.ignore
    def get_classifier(self):
        """Get the classification head module.

        Returns:
            nn.Module: The classification head (Linear layer or Identity).
        """
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        """Reset the classification head with a new number of classes.

        Args:
            num_classes (int): New number of classes for classification.
            global_pool (str, optional): Global pooling type (unused in current implementation).
                Defaults to "".
        """
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def encode_text(self, inputs: Dict[str, torch.Tensor], normalize: bool = False):
        output = self.inner.text_model(**inputs, return_dict=True)
        token = output.pooler_output

        if normalize:
            token = F.normalize(token, dim=-1)

        return token

    def zero_shot_postproc(self, logits: torch.Tensor):
        logit_scale, logit_bias = self.inner.logit_scale.to(logits.device), self.inner.logit_bias.to(logits.device)
        logits = logits * logit_scale.exp() + logit_bias
        return logits


class WrappedTokenizer:
    def __init__(self, proc):
        self._proc = proc

    def __call__(self, text: List[str]):
        c_text = [canonicalize_text(t) for t in text]
        return self._proc(text=c_text, return_tensors='pt', max_length=64, padding='max_length', truncation=True)


def canonicalize_text(
    text: str,
    *,
    keep_punctuation_exact_string=None,
    trans_punctuation: dict = str.maketrans("", "", string.punctuation),
):
    """Returns canonicalized `text` (lowercase and punctuation removed).

    From: https://github.com/google-research/big_vision/blob/53f18caf27a9419231bbf08d3388b07671616d3d/big_vision/evaluators/proj/image_text/prompt_engineering.py#L94

    Args:
      text: string to be canonicalized.
      keep_punctuation_exact_string: If provided, then this exact string kept.
        For example providing '{}' will keep any occurrences of '{}' (but will
        still remove '{' and '}' that appear separately).
    """
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(trans_punctuation)
            for part in text.split(keep_punctuation_exact_string)
        )
    else:
        text = text.translate(trans_punctuation)
    text = text.lower()
    text = " ".join(text.split())
    return text.strip()


def get_siglip2_model(version: str, hf_model_path: str = None):
    """Load SigLIP2 model from HuggingFace Hub or local path.
    
    Args:
        version: Model version identifier
        hf_model_path: Optional local path to load model from. If provided,
                       loads from local path instead of HF Hub.
    """
    version_map = {
        'siglip2-so400m-512': ('google/siglip2-so400m-patch16-512', False, 16),
        'siglip2-so400m': ('google/siglip2-so400m-patch16-naflex', True, 16),
        'siglip2-g-384': ('google/siglip2-giant-opt-patch16-384', False, 16),
    }
    version_map['siglip2'] = version_map['siglip2-so400m']
    version_map['siglip2-g'] = version_map['siglip2-g-384']

    version, is_dynamic, patch_size = version_map[version]

    # Load from local path if provided, otherwise from HF Hub
    if hf_model_path:
        print(f"Loading SigLIP2 from local path: {hf_model_path}")
        model = AutoModel.from_pretrained(hf_model_path, trust_remote_code=True, local_files_only=True)
        proc = AutoProcessor.from_pretrained(hf_model_path, trust_remote_code=True, local_files_only=True)
    else:
        print(f"Loading SigLIP2 from HuggingFace Hub: {version}")
        model = AutoModel.from_pretrained(version, trust_remote_code=True)
        proc = AutoProcessor.from_pretrained(version, trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained(version, trust_remote_code=True)

    img_proc = proc.image_processor

    tokenizer = WrappedTokenizer(proc)

    model = SigLIP2Wrapper(model, tokenizer, num_classes=0, is_dynamic=is_dynamic, patch_size=patch_size)

    return model


@BACKBONE_REGISTRY.register()
def siglip2_so400m_patch16_512(hf_model_path: str = None, **kwargs):
    """SigLIP2 SO400M Patch16 512."""
    return get_siglip2_model("siglip2-so400m-512", hf_model_path=hf_model_path)


@BACKBONE_REGISTRY.register()
def siglip2_so400m(hf_model_path: str = None, **kwargs):
    """SigLIP2 SO400M."""
    return get_siglip2_model("siglip2-so400m", hf_model_path=hf_model_path)


@BACKBONE_REGISTRY.register()
def siglip2_g_384(hf_model_path: str = None, **kwargs):
    """SigLIP2 Giant 384."""
    return get_siglip2_model("siglip2-g-384", hf_model_path=hf_model_path)
