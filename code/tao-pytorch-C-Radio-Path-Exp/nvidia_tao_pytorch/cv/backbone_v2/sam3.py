
from typing import Tuple, Union, Any

from einops import rearrange
import torch
from torch import nn

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase

try:
    from sam3.model_builder import build_sam3_image_model
except Exception as e:
    print(f'Failed to import SAM3. Error: {e}')
    build_sam3_image_model = None
norm_t = Union[Tuple[float, float, float], torch.Tensor]


def _to_tensor(v: norm_t):
    return torch.as_tensor(v, dtype=torch.float32).view(-1, 1, 1)


class InputConditioner(nn.Module):
    def __init__(self,
                 input_scale: float,
                 norm_mean: norm_t,
                 norm_std: norm_t,
                 dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.dtype = dtype

        # self.input_scale = input_scale
        self.register_buffer("norm_mean", _to_tensor(norm_mean) / input_scale)
        self.register_buffer("norm_std", _to_tensor(norm_std) / input_scale)

    def forward(self, x: torch.Tensor):
        # x = x * self.input_scale
        y = (x - self.norm_mean) / self.norm_std
        return y.to(self.dtype)

    def backward(self, x: torch.Tensor):
        y = x * self.norm_std + self.norm_mean
        return y.to(self.dtype)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)

        # If changing dtype, change self.dtype to match
        dtype_kwarg = kwargs.get('dtype', None)
        if dtype_kwarg is not None:
            self.dtype = dtype_kwarg
        else:
            dtype_args = [arg for arg in args if isinstance(arg, torch.dtype)]
            if len(dtype_args) == 1:
                self.dtype = dtype_args[0]

        return self

    def float(self):
        super().float()
        self.dtype = torch.float
        return self

    def double(self):
        super().double()
        self.dtype = torch.double
        return self

    def half(self):
        super().half()
        self.dtype = torch.half
        return self

    def bfloat16(self):
        super().bfloat16()
        self.dtype = torch.bfloat16
        return self

    @staticmethod
    def default():
        from timm.data.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

        return InputConditioner(
            input_scale=1.0,
            norm_mean=OPENAI_CLIP_MEAN,
            norm_std=OPENAI_CLIP_STD,
        )


class SAM3Wrapper(BackboneBase):
    def __init__(self,
                 sam3_vision_encoder,
                 num_classes: int = 0,
                 in_chans: int = 3,
                 activation_checkpoint=False,
                 freeze_at=None,
                 freeze_norm=False,
                 head_init_scale=1.0,
                 **kwargs,):
        super().__init__(
            in_chans=in_chans,
            num_classes=num_classes,
            activation_checkpoint=activation_checkpoint,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
        )

        # Extract the ViT trunk directly from the vision backbone (Sam3DualViTDetNeck)
        # This gets us features before the neck applies dimensional bottleneck
        self.inner = sam3_vision_encoder.trunk
        self.num_features = self.inner.patch_embed.proj.out_channels
        if num_classes > 0:
            self.head = nn.Linear(self.num_features, num_classes)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)
        else:
            self.head = nn.Identity()

    @property
    def embed_dim(self):
        return self.inner.patch_embed.proj.out_channels

    @property
    def patch_size(self):
        return self.inner.patch_embed.proj.stride[0]

    def get_stage_dict(self):
        """Get the stage dictionary."""
        stage_dict = {}
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

    @torch.no_grad()
    def forward(self, x: torch.Tensor, return_features: bool = False, return_logits: bool = False):
        # Run the ViT trunk directly to get features before dimensional bottleneck
        # The trunk returns a list of outputs from global attention blocks
        # (typically just the final output unless return_interm_layers is True)
        features_list = self.inner(x)

        # Extract the last feature tensor (final global attention output)
        # Features are in [B, C, H, W] format from SAM3's ViT output
        features = features_list[-1]

        # Reshape from [B, C, H, W] to [B, H*W, C] for consistency with other teachers
        # features = rearrange(features, "b c h w -> b (h w) c")
        # summary = features.mean(dim=1)
        summary = features.mean(dim=(2, 3))
        if return_features:
            return summary, features
        else:
            if return_logits:
                return summary
            else:
                return self.head(summary)


"""
    sam3_model, input_conditioner = get_sam3_model(model, chk_base_path=chk_base_path, load_from_HF=load_from_HF, wrap=False)
    img_encoder = SAM3Wrapper(sam3_model.backbone.vision_backbone)
"""

def get_sam3_model(chk_base_path: str = None,
                   wrap: bool = True,
                   load_from_HF: bool = True,
                   **kwargs) -> Tuple[Union[SAM3Wrapper, Any], InputConditioner]:
    if build_sam3_image_model is None:
        raise ImportError(f'Unable to import Sam3 module. Please install: pip install git+https://github.com/facebookresearch/sam3.git')

    # SAM3 uses HuggingFace by default, can override with checkpoint_path
    if chk_base_path and not load_from_HF:
        checkpoint_path = chk_base_path
    else:
        checkpoint_path = None  # Will download from HuggingFace

    # Build the SAM3 image model
    # This returns a Sam3Image model with a backbone (SAM3VLBackbone)
    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path,
        load_from_HF=load_from_HF,
        eval_mode=True,
        device='cuda',
        enable_inst_interactivity=False,  # Don't need SAM1 task for teacher
        **kwargs
    )

    # SAM3 uses 0.5 mean/std normalization (not ImageNet stats)
    # See: https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_image_processor.py
    conditioner = InputConditioner(
        input_scale=1.0,
        norm_mean=[0.5, 0.5, 0.5],
        norm_std=[0.5, 0.5, 0.5],
    )

    if not wrap:
        return model, conditioner

    # Extract the vision encoder (Sam3DualViTDetNeck) from the backbone
    # model.backbone is SAM3VLBackbone, which has vision_backbone as Sam3DualViTDetNeck
    vision_encoder = model.backbone.vision_backbone
    img_encoder = SAM3Wrapper(vision_encoder)

    return img_encoder


@BACKBONE_REGISTRY.register()
def sam3_default(**kwargs):
    chk_base_path = "<NVR_ROOT>/users/<user>/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
    return get_sam3_model(chk_base_path=chk_base_path, wrap=True, load_from_HF=False)
