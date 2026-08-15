from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class GradientLoss(nn.Module):
    """Charbonnier distance between Sobel gradients."""

    def __init__(self, epsilon: float = 1e-3):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])[None, None] / 8.0,
        )
        self.register_buffer("sobel_y", self.sobel_x.transpose(-1, -2).contiguous())

    def _gradient(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        channels = image.shape[1]
        kernel_x = self.sobel_x.expand(channels, 1, 3, 3)
        kernel_y = self.sobel_y.expand(channels, 1, 3, 3)
        return (
            F.conv2d(image, kernel_x, padding=1, groups=channels),
            F.conv2d(image, kernel_y, padding=1, groups=channels),
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_x, pred_y = self._gradient(prediction.float())
        target_x, target_y = self._gradient(target.float())
        difference = (pred_x - target_x).square() + (pred_y - target_y).square()
        return torch.sqrt(difference + self.epsilon**2).mean()
