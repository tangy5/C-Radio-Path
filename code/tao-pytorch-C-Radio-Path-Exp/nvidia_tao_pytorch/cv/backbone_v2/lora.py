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

"""LoRA (Low-Rank Adaptation) for RADIO ViT backbones.

Injects trainable low-rank matrices into targeted linear layers while freezing
all original backbone weights. This preserves pretrained embeddings while
allowing domain-specific adaptation during distillation training.
"""

import math
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """Wraps an nn.Linear with a low-rank additive adaptation.

    The forward pass computes: output = original_linear(x) + (alpha/r) * x @ A^T @ B^T
    At initialization, B is zero so the LoRA contribution is zero and the
    pretrained output is preserved exactly.
    """

    def __init__(self, original_linear, r=16, alpha=32, dropout=0.0):
        super().__init__()
        self.linear = original_linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        result = self.linear(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result + lora_out * self.scaling

    def merge_weights(self):
        """Merge LoRA weights into the original linear layer for inference."""
        self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
        self.lora_A.data.zero_()
        self.lora_B.data.zero_()

    def extra_repr(self):
        return (
            f"in_features={self.linear.in_features}, "
            f"out_features={self.linear.out_features}, "
            f"r={self.r}, alpha={self.alpha}, "
            f"bias={self.linear.bias is not None}"
        )


def apply_lora_to_radio(model, config):
    """Inject LoRA adapters into a RADIO model's ViT blocks.

    Navigates the model hierarchy: model.radio.radio.model.blocks[i]
    and replaces target nn.Linear layers with LoRALinear wrappers.

    Args:
        model: The RADIO model instance.
        config: LoRA config with attributes: r, alpha, dropout,
                target_modules (list of str), block_range (str or list).
    """
    r = config.r
    alpha = config.alpha
    dropout = getattr(config, 'dropout', 0.0)
    target_modules = list(config.target_modules)
    block_range = getattr(config, 'block_range', 'all')

    # Navigate to timm VisionTransformer blocks
    vit_blocks = model.radio.radio.model.blocks
    num_blocks = len(vit_blocks)

    if block_range == 'all':
        target_indices = range(num_blocks)
    elif isinstance(block_range, str) and block_range.startswith('last_'):
        n = int(block_range.split('_')[1])
        target_indices = range(num_blocks - n, num_blocks)
    else:
        target_indices = range(num_blocks)

    # Map target names to (parent_module, attribute_name)
    module_map = {
        'qkv': lambda b: (b.attn, 'qkv'),
        'proj': lambda b: (b.attn, 'proj'),
        'fc1': lambda b: (b.mlp, 'fc1'),
        'fc2': lambda b: (b.mlp, 'fc2'),
    }

    count = 0
    for idx in target_indices:
        block = vit_blocks[idx]
        for target_name in target_modules:
            if target_name not in module_map:
                raise ValueError(
                    f"Unknown LoRA target module '{target_name}'. "
                    f"Choose from: {list(module_map.keys())}"
                )
            parent, attr_name = module_map[target_name](block)
            original_linear = getattr(parent, attr_name)

            if not isinstance(original_linear, nn.Linear):
                raise TypeError(
                    f"Expected nn.Linear for {target_name} in block {idx}, "
                    f"got {type(original_linear)}"
                )

            lora_layer = LoRALinear(
                original_linear, r=r, alpha=alpha, dropout=dropout
            )
            setattr(parent, attr_name, lora_layer)
            count += 1

            logger.info(
                f"LoRA: block {idx:>2d}.{target_name:<4s} "
                f"({original_linear.in_features} -> {original_linear.out_features})"
            )

    logger.info(
        f"LoRA applied: {count} adapters across {len(list(target_indices))} blocks "
        f"(r={r}, alpha={alpha}, targets={target_modules})"
    )
    return model


def freeze_non_lora_params(model):
    """Freeze all parameters except LoRA (lora_A, lora_B) in the model.

    This should be called after apply_lora_to_radio() and after any other
    trainable modules (like MLP heads) have been set up.
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze LoRA parameters
    lora_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
            lora_count += 1

    logger.info(f"Frozen all params, unfrozen {lora_count} LoRA adapter pairs")
    return model


def get_lora_state_dict(model):
    """Return a state dict containing only LoRA parameters."""
    lora_params = {}
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_params[name] = param.data.clone()
    return lora_params


def merge_lora_weights(model):
    """Merge all LoRA weights into the original linear layers.

    After calling this, the LoRA contribution is absorbed into the
    original weights and the LoRALinear wrappers can optionally be
    removed. Useful for inference/export.
    """
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()
            count += 1
    logger.info(f"Merged {count} LoRA adapters into original weights")
    return model


def print_lora_summary(model):
    """Print a summary of LoRA vs total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = 0
    for name, param in model.named_parameters():
        if ('lora_A' in name or 'lora_B' in name) and param.requires_grad:
            lora_params += param.numel()

    logger.info(
        f"Parameter summary: "
        f"total={total:,}  "
        f"trainable={trainable:,} ({100*trainable/total:.2f}%)  "
        f"lora={lora_params:,} ({100*lora_params/total:.2f}%)"
    )
