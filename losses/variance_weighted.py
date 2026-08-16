from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SmoothRegionVarianceLoss(nn.Module):
    """Penalize residual grain in smooth GT background regions using local variance masking."""

    def __init__(self, kernel_size: int = 5, variance_threshold: float = 0.005, boost_factor: float = 2.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.variance_threshold = variance_threshold
        self.boost_factor = boost_factor

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        padding = self.kernel_size // 2
        pred_flt = prediction.float()
        target_flt = target.float()

        # Compute local variance of GT using sliding box average
        mean_gt = F.avg_pool2d(target_flt, kernel_size=self.kernel_size, stride=1, padding=padding)
        sq_mean_gt = F.avg_pool2d(target_flt ** 2, kernel_size=self.kernel_size, stride=1, padding=padding)
        var_gt = (sq_mean_gt - mean_gt ** 2).clamp(min=0.0)

        # Smooth regions have GT variance below threshold
        smooth_mask = (var_gt < self.variance_threshold).float()

        # Base Charbonnier difference
        diff = torch.sqrt((pred_flt - target_flt) ** 2 + 1e-6)

        # Apply boost multiplier in smooth background regions
        weights = 1.0 + (self.boost_factor - 1.0) * smooth_mask
        return (diff * weights).mean()
