"""Paired-data augmentation that matches the released mixed-degradation task."""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F


class MixedDegradationAugmentor:
    """Generate an additional LR observation from a clean HR target.

    Gaussian blur, 2x reduction, multiplicative speckle, and additive Gaussian
    noise are sampled independently. Noise may occur before or after reduction
    because the official task states that degradation order is not fixed.
    """

    def __init__(self, probability: float = 0.35):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        self.probability = probability

    @staticmethod
    def _blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma <= 0.0:
            return image
        radius = 3
        coordinates = torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
        kernel_1d = torch.exp(-(coordinates.square()) / (2.0 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel = (kernel_1d[:, None] * kernel_1d[None, :]).view(1, 1, 7, 7)
        padded = F.pad(image.unsqueeze(0), (radius,) * 4, mode="reflect")
        return F.conv2d(padded, kernel).squeeze(0)

    @staticmethod
    def _noise(image: torch.Tensor, speckle_sigma: float, gaussian_sigma: float) -> torch.Tensor:
        if speckle_sigma:
            image = image + image * torch.randn_like(image) * speckle_sigma
        if gaussian_sigma:
            image = image + torch.randn_like(image) * gaussian_sigma
        return image

    @staticmethod
    def _downsample(image: torch.Tensor, mode: str, target_size: tuple[int, int] = (128, 128)) -> torch.Tensor:
        kwargs: dict[str, object] = {"size": target_size, "mode": mode}
        if mode in {"bilinear", "bicubic"}:
            kwargs.update(align_corners=False, antialias=True)
        return F.interpolate(image.unsqueeze(0), **kwargs).squeeze(0)

    def __call__(self, clean_hr: torch.Tensor, official_lr: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.probability:
            return official_lr

        target_size = (official_lr.shape[-2], official_lr.shape[-1])
        sigma = random.uniform(0.2, 1.6) if random.random() < 0.7 else 0.0
        speckle_sigma = random.uniform(0.015, 0.18) if random.random() < 0.9 else 0.0
        gaussian_sigma = random.uniform(0.003, 0.08) if random.random() < 0.9 else 0.0
        mode = random.choice(("area", "bilinear", "bicubic"))

        degraded = self._blur(clean_hr, sigma)
        if random.random() < 0.5:
            degraded = self._noise(degraded, speckle_sigma, gaussian_sigma)
            degraded = self._downsample(degraded, mode, target_size)
        else:
            degraded = self._downsample(degraded, mode, target_size)
            degraded = self._noise(degraded, speckle_sigma, gaussian_sigma)

        # Match the released LR distribution without clipping its informative tails.
        contrast = random.uniform(0.92, 1.08)
        offset = random.uniform(-0.02, 0.02)
        return degraded * contrast + offset
