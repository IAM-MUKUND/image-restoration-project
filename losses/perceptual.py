"""Differentiable LPIPS loss for restoration fine-tuning."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class PerceptualLoss(nn.Module):
    """Frozen LPIPS-AlexNet with gradients retained for the prediction."""

    def __init__(self, max_size: int = 128):
        super().__init__()
        import lpips

        self.model = lpips.LPIPS(net="alex").eval()
        self.model.requires_grad_(False)
        self.max_size = max_size

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = prediction.float().clamp(0.0, 1.0)
        target = target.float().clamp(0.0, 1.0)
        if max(prediction.shape[-2:]) > self.max_size:
            size = (self.max_size, self.max_size)
            prediction = F.interpolate(
                prediction, size=size, mode="bilinear", align_corners=False, antialias=True
            )
            target = F.interpolate(
                target, size=size, mode="bilinear", align_corners=False, antialias=True
            )
        prediction = prediction.repeat(1, 3, 1, 1) * 2.0 - 1.0
        target = target.repeat(1, 3, 1, 1) * 2.0 - 1.0
        return self.model(prediction, target, normalize=False).mean()
