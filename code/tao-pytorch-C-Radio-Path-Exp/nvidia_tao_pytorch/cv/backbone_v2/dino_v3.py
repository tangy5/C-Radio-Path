import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
import timm

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase


class DINOV3Wrapper(BackboneBase):
    def __init__(self,
                 dino_model: nn.Module,
                 num_classes: int = 0,
                 in_chans: int = 3,
                 activation_checkpoint=False,
                 freeze_at=None,
                 freeze_norm=False,
                 head_init_scale=1.0,
                 **kwargs) -> None:
        super().__init__(
            in_chans=in_chans,
            num_classes=num_classes,
            activation_checkpoint=activation_checkpoint,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
        )
        self.inner = dino_model
        self.num_features = self.inner.num_features
        if num_classes > 0:
            self.head = nn.Linear(self.num_features, num_classes)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)
        else:
            self.head = nn.Identity()

    @property
    def embed_dim(self):
        return self.inner.embed_dim

    @property
    def patch_size(self):
        return 16

    @property
    def num_summary_tokens(self):
        return self.inner.num_prefix_tokens

    def get_stage_dict(self):
        """Get the stage dictionary."""
        stage_dict = {0: self.inner.patch_embed}
        for i, block in enumerate(self.inner.blocks, start=1):
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


    def forward_feature_pyramid(self, x: torch.Tensor):
        _, features = self.forward(x, return_features=True, return_logits=False)
        return features
    
    def forward(self, x: torch.Tensor, return_features: bool = False, return_logits: bool = False):
        B, _, height, width = x.shape
        x = self.inner.forward_features(x)

        cls_token = x[:, 0]
        features = x[:, self.inner.num_prefix_tokens:]

        if return_features:
            # reshape to BCHW output format
            H, W = self.inner.patch_embed.dynamic_feat_size((height, width))
            features = features.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            # features = rearrange(features, 'b (h w) c -> b c h w', h=out_h, w=out_w)
            return cls_token, features
        else:
            if return_logits:
                return cls_token
            else:
                return self.head(cls_token)


_DEFAULT_BASE_PATH = '<EDGEAI_PROJECT_DIR>/users/<user>/experiments/dinov3'
_CHECKPOINT_MAP = {
    'dinov3_vit7b16': 'dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth',
    'dinov3_vith16plus': 'dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth',
    'dinov3_vitl16': 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
    'dinov3_vitb16': 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
}

def _load_dino_v3(dino_v3_model, is_local: bool = False, hf_model_path: str = None):
    # option1: use torch.hub
    # model = torch.hub.load(
    #     'facebookresearch/dinov3',
    #     dino_v3_model,
    #     pretrained=True,
    # )
    # if pretrained:
    #     chk_path = os.path.join(base_path, _CHECKPOINT_MAP[dino_v3_model])
    #     chk = torch.load(chk_path, map_location='cpu')
    #     model.load_state_dict(chk)
    # return model
    # option2: use transformers
    # if is_local:
    #     ckpt_path = os.path.join(_DEFAULT_BASE_PATH, dino_v3_model)
    # else:
    #     ckpt_path = dino_v3_model
    # print(f"Loading DINOv3 model from {ckpt_path}")
    # model = AutoModel.from_pretrained(ckpt_path)
    # option3: use timm
    if hf_model_path:
        # Load from local checkpoint file
        print(f"Loading DINOv3 from local checkpoint: {hf_model_path}")
        model = timm.create_model(dino_v3_model, pretrained=False, checkpoint_path=hf_model_path)
    elif is_local:
        # Load from local default path
        ckpt_path = os.path.join(_DEFAULT_BASE_PATH, _CHECKPOINT_MAP.get(dino_v3_model, ''))
        print(f"Loading DINOv3 from local default path: {ckpt_path}")
        model = timm.create_model(dino_v3_model, pretrained=False, checkpoint_path=ckpt_path)
    else:
        # Load from HuggingFace Hub
        print(f"Loading DINOv3 from HuggingFace Hub: {dino_v3_model}")
        model = timm.create_model(dino_v3_model, pretrained=True)
    return model


@BACKBONE_REGISTRY.register()
def dinov3_vit7b16(hf_model_path: str = None, **kwargs):
    model = _load_dino_v3("vit_7b_patch16_dinov3.lvd1689m", is_local=False, hf_model_path=hf_model_path)
    model = DINOV3Wrapper(model, **kwargs)
    return model


@BACKBONE_REGISTRY.register()
def dinov3_vitl16(hf_model_path: str = None, **kwargs):
    model = _load_dino_v3("vit_large_patch16_dinov3.lvd1689m", is_local=False, hf_model_path=hf_model_path)
    model = DINOV3Wrapper(model, **kwargs)
    return model


@BACKBONE_REGISTRY.register()
def dinov3_vitb16(hf_model_path: str = None, **kwargs):
    model = _load_dino_v3("vit_base_patch16_dinov3.lvd1689m", is_local=False, hf_model_path=hf_model_path)
    model = DINOV3Wrapper(model, **kwargs)
    return model


@BACKBONE_REGISTRY.register()
def dinov3_vith16plus(hf_model_path: str = None, **kwargs):
    model = _load_dino_v3("vit_huge_plus_patch16_dinov3.lvd1689m", is_local=False, hf_model_path=hf_model_path)
    model = DINOV3Wrapper(model, **kwargs)
    return model