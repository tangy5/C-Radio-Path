# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""
GPU-batched image augmentations.

Ported from EVFM. Enables batched, GPU-accelerated color and non-geometric
augmentations. Uses color_matrix.py for 4x4 color transforms (no adlr_ops).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli, Binomial, Normal, Uniform

from nvidia_tao_pytorch.cv.classification_pyt.dataloader.color_matrix import (
    apply_color_transform,
    contrast_matrix,
    hue_saturation_matrix,
)


def _get_device_dtype(tensor):
    """Get device and dtype from tensor for creating compatible tensors."""
    return tensor.device, tensor.dtype


class AugmentationBase:
    """Base class for GPU-batched data augmentations."""

    def __init__(self, prob_apply=0.5):
        if prob_apply < 0 or prob_apply > 1:
            raise ValueError("`prob_apply` must be between 0 and 1 inclusive.")
        self.prob_apply = prob_apply
        self.num_image_distribution = None

    def __call__(self, images, *additional):
        if images.shape[0] == 0:
            return images

        if (self.num_image_distribution is None or
                self.num_image_distribution.total_count != images.shape[0]):
            probs = torch.tensor([self.prob_apply], device=images.device)
            self.num_image_distribution = Binomial(
                total_count=images.shape[0], probs=probs
            )

        num_images = int(self.num_image_distribution.sample().item())
        if num_images == 0:
            return images

        if num_images < images.shape[0]:
            offset = np.random.randint(0, images.shape[0] - num_images)
        else:
            offset = 0

        sub_images = images[offset:offset + num_images]
        applied = self.get_applied_images(sub_images, *additional)
        return torch.cat([
            images[:offset], applied, images[offset + num_images:]
        ], dim=0)

    def get_applied_images(self, images, *additional):
        raise NotImplementedError("Subclasses must implement this!")


class GaussianBlurAugmentation(AugmentationBase):
    """Applies Gaussian blurring with the specified kernel size."""

    def __init__(self, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd and >= 3.")

        kernel = torch.zeros((kernel_size,), dtype=torch.float32)
        kernel[0] = 1
        for _ in range(1, kernel_size):
            for i in range(kernel_size - 1, 0, -1):
                pv = kernel[i - 1] if i > 0 else 0
                kernel[i] = pv + kernel[i]

        k = kernel.reshape(kernel_size, 1)
        kernel = (k @ kernel.reshape(1, kernel_size))
        kernel /= kernel.sum()
        self.weight = kernel.reshape(1, 1, kernel_size, kernel_size)
        self.padding = kernel_size // 2

    def get_applied_images(self, images, *additional):
        weight = self.weight.to(images)
        s_images = images.reshape(-1, 1, *images.shape[-2:])
        was_bench = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False
        blurred = F.conv2d(s_images, weight, padding=(self.padding,) * 2)
        torch.backends.cudnn.benchmark = was_bench
        return blurred.reshape(*images.shape)


class BlurAugmentation(AugmentationBase):
    """Applies box (uniform) blur."""

    def __init__(self, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd and >= 3.")
        kernel = torch.full(
            (kernel_size, kernel_size), 1.0 / (kernel_size ** 2),
            dtype=torch.float32
        )
        self.weight = kernel.reshape(1, 1, kernel_size, kernel_size)
        self.padding = kernel_size // 2

    def get_applied_images(self, images, *additional):
        weight = self.weight.to(images)
        s_images = images.reshape(-1, 1, *images.shape[-2:])
        padded = F.pad(s_images, (self.padding,) * 4, mode='reflect')
        blurred = F.conv2d(padded, weight)
        return blurred.reshape(*images.shape)


class GrayscaleAugmentation(AugmentationBase):
    """Converts to grayscale by averaging channels."""

    def get_applied_images(self, images, *additional):
        mean = torch.mean(images, dim=1, keepdim=True)
        return mean.expand_as(images)


class RandomNoiseAugmentation(AugmentationBase):
    """Applies random Gaussian noise."""

    def __init__(self, std_dev, mean=0.0, **kwargs):
        super().__init__(**kwargs)
        self.variance = std_dev ** 2
        self.mean = mean

    def get_applied_images(self, images, *additional):
        B = images.shape[0]
        applied_var = torch.rand(
            (B, 1, 1, 1), device=images.device, dtype=images.dtype
        ) * self.variance
        noise = torch.randn_like(images, device=images.device) * applied_var
        noise = noise + self.mean
        return images + noise


class ColorAugmentation:
    """Base class for color augmentations using 4x4 matrices."""

    def __init__(self, clamp_min=0.0, clamp_max=1.0, prob_apply=0.5,
                 verbose=False):
        if prob_apply < 0 or prob_apply > 1:
            raise ValueError("`prob_apply` must be between 0 and 1.")
        if clamp_max <= clamp_min:
            raise ValueError("clamp_max must be > clamp_min.")
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.prob_apply = prob_apply
        self.verbose = verbose
        self.identity = None

    def __call__(self, images, *additional):
        transforms = self.get_transform_matrices(
            images.shape[0], images.device
        )
        transforms = transforms.to(images.device, non_blocking=True)
        result = apply_color_transform(
            images, transforms, self.clamp_min, self.clamp_max
        )
        return result

    def get_transform_matrices(self, num_images, device=None):
        if num_images < 1:
            raise ValueError("num_images must be >= 1.")

        if device is None:
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            device = torch.device(dev)

        if self.identity is None or self.identity.shape[0] < num_images:
            self.identity = torch.eye(
                4, dtype=torch.float32, device=device
            ).reshape(1, 4, 4).repeat(num_images, 1, 1)
        identity = self.identity[:num_images].to(device)

        sample_mask = Bernoulli(
            torch.tensor([self.prob_apply], device=device)
        ).sample((num_images,)).reshape(-1, 1, 1)

        inner = self.get_applied_matrices(num_images, device)
        if inner is not None:
            transforms = sample_mask * inner + (1 - sample_mask) * identity
        else:
            transforms = identity
        return transforms

    def get_applied_matrices(self, num_images, device=None):
        raise NotImplementedError("Subclasses must implement this!")


class BrightnessAugmentation(ColorAugmentation):
    """Random brightness (additive offset)."""

    def __init__(self, scale_max=0.5, abs_max=0.4, **kwargs):
        super().__init__(**kwargs)
        self.scale_max = scale_max
        self.abs_max = abs_max

    def get_applied_matrices(self, num_images, device=None):
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        dist = Normal(
            torch.tensor([0.0], device=device),
            torch.tensor([self.scale_max / 2.0], device=device)
        )
        offsets = dist.sample((num_images,)).clamp(
            min=-self.abs_max, max=self.abs_max
        )
        identity = torch.eye(
            4, dtype=torch.float32, device=device
        ).reshape(1, 4, 4).repeat(num_images, 1, 1)
        offsets = offsets.repeat(1, 4)
        offsets[:, 3] = 1
        identity[:, -1] = offsets
        return identity


class HueAugmentation(ColorAugmentation):
    """Random hue and saturation."""

    def __init__(self, hue_rotation_max=50.0, saturation_shift_max=0.4,
                 **kwargs):
        super().__init__(**kwargs)
        self.hue_rotation_max = hue_rotation_max
        self.saturation_shift_max = saturation_shift_max

    def get_applied_matrices(self, num_images, device=None):
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        hue_scale = max(self.hue_rotation_max / 2.0, 1e-6)  # Normal requires scale > 0
        hue_dist = Normal(
            torch.tensor([0.0], device=device),
            torch.tensor([hue_scale], device=device)
        )
        sat_dist = Uniform(
            torch.tensor([-self.saturation_shift_max], device=device),
            torch.tensor([self.saturation_shift_max], device=device)
        )
        hue = hue_dist.sample((num_images,)).reshape(-1)
        sat = 1.0 + sat_dist.sample((num_images,)).reshape(-1)
        return hue_saturation_matrix(hue, sat)


class ContrastAugmentation(ColorAugmentation):
    """Random contrast."""

    def __init__(self, scale_max=0.5, center=0.5, **kwargs):
        super().__init__(**kwargs)
        self.scale_max = scale_max
        self.center = center

    def get_applied_matrices(self, num_images, device=None):
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        scale = max(self.scale_max / 2.0, 1e-6)  # Normal requires scale > 0
        dist = Normal(
            torch.tensor([0.0], device=device),
            torch.tensor([scale], device=device)
        )
        contrast = dist.sample((num_images,)).reshape(-1)
        center = torch.tensor(
            self.center, dtype=torch.float32, device=contrast.device
        )
        return contrast_matrix(contrast, center)


class CompositeAugmentation:
    """Chains multiple augmentations."""

    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, images, *additional):
        for aug in self.augmentations:
            images = aug(images, *additional)
        return images


class CompositeColorAugmentation(ColorAugmentation):
    """Composite of multiple color augmentations (fused matrix multiply)."""

    def __init__(self, augmentations, **kwargs):
        kwargs['prob_apply'] = 1.0
        super().__init__(**kwargs)
        self.augmentations = augmentations

    def get_transform_matrices(self, num_images, device=None):
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        transform = torch.eye(
            4, dtype=torch.float32, device=device
        ).reshape(1, 4, 4).repeat(num_images, 1, 1)
        for aug in self.augmentations:
            aug_mat = aug.get_transform_matrices(num_images, device)
            transform = torch.bmm(transform, aug_mat)
        sample_mask = Bernoulli(
            torch.tensor([self.prob_apply], device=device)
        ).sample((num_images,)).reshape(-1, 1, 1)
        identity = torch.eye(
            4, dtype=torch.float32, device=device
        ).reshape(1, 4, 4).repeat(num_images, 1, 1)
        return sample_mask * transform + (1 - sample_mask) * identity

    def get_applied_matrices(self, num_images, device=None):
        return None


def create_gpu_color_augmentations(
    brightness=0.4,
    contrast=0.4,
    hue=0.4,
    saturation=0.4,
    blur_prob=0.1,
    noise_std=0.02,
    grayscale_prob=0.0,
    prob_apply=0.5,
    clamp_min=0.0,
    clamp_max=1.0,
):
    """
    Create a composite GPU color augmentation pipeline.

    Args:
        brightness: Brightness scale max (additive offset).
        contrast: Contrast scale max.
        hue: Hue rotation max in degrees.
        saturation: Saturation shift max (multiplier around 1.0).
        blur_prob: Probability of Gaussian blur.
        noise_std: Std dev for random noise.
        grayscale_prob: Probability of grayscale.
        prob_apply: Per-image apply probability.
        clamp_min: Min clamp value.
        clamp_max: Max clamp value.

    Returns:
        CompositeAugmentation or CompositeColorAugmentation.
    """
    augs = []

    if brightness > 0:
        augs.append(BrightnessAugmentation(
            scale_max=brightness / 2.0, abs_max=brightness,
            prob_apply=prob_apply, clamp_min=clamp_min, clamp_max=clamp_max
        ))
    if contrast > 0:
        augs.append(ContrastAugmentation(
            scale_max=contrast, center=0.5,
            prob_apply=prob_apply, clamp_min=clamp_min, clamp_max=clamp_max
        ))
    if hue > 0 or saturation > 0:
        augs.append(HueAugmentation(
            hue_rotation_max=hue, saturation_shift_max=saturation,
            prob_apply=prob_apply, clamp_min=clamp_min, clamp_max=clamp_max
        ))
    if blur_prob > 0:
        augs.append(GaussianBlurAugmentation(
            kernel_size=5, prob_apply=blur_prob
        ))
    if noise_std > 0:
        augs.append(RandomNoiseAugmentation(
            std_dev=noise_std, prob_apply=prob_apply
        ))
    if grayscale_prob > 0:
        augs.append(GrayscaleAugmentation(prob_apply=grayscale_prob))

    if len(augs) == 0:
        return None
    if len(augs) == 1:
        return augs[0]
    return CompositeAugmentation(augs)
