from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .frequency import FrequencyLoss
from .gradient import GradientLoss
from .l1 import CharbonnierLoss
from .ssim import SSIMLoss


class CombinedRestorationLoss(nn.Module):
    def __init__(
        self,
        ssim_weight: float = 0.1,
        frequency_weight: float = 0.0,
        gradient_weight: float = 0.0,
        auxiliary_weight: float = 0.0,
        uncertainty_weight: float = 0.0,
    ):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.frequency_weight = frequency_weight
        self.gradient_weight = gradient_weight
        self.auxiliary_weight = auxiliary_weight
        self.uncertainty_weight = uncertainty_weight
        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.frequency = FrequencyLoss()
        self.gradient = GradientLoss()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        clean_lr: torch.Tensor | None = None,
        uncertainty: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = self.charbonnier(prediction, target)
        if self.ssim_weight:
            loss = loss + self.ssim_weight * self.ssim(prediction, target)
        if self.frequency_weight:
            loss = loss + self.frequency_weight * self.frequency(prediction, target)
        if self.gradient_weight:
            loss = loss + self.gradient_weight * self.gradient(prediction, target)
        if self.auxiliary_weight and clean_lr is not None:
            clean_lr_target = F.interpolate(
                target.float(), size=clean_lr.shape[-2:], mode="bicubic", align_corners=False, antialias=True
            )
            loss = loss + self.auxiliary_weight * self.charbonnier(clean_lr, clean_lr_target)
        if self.uncertainty_weight and uncertainty is not None:
            error_target = (prediction.detach().float() - target.float()).abs()
            loss = loss + self.uncertainty_weight * F.smooth_l1_loss(
                uncertainty.float(), error_target
            )
        return loss
