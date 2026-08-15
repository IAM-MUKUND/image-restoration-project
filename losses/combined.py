from __future__ import annotations

import torch
from torch import nn

from .frequency import FrequencyLoss
from .l1 import CharbonnierLoss
from .ssim import SSIMLoss


class CombinedRestorationLoss(nn.Module):
    def __init__(self, ssim_weight: float = 0.1, frequency_weight: float = 0.0):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.frequency_weight = frequency_weight
        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.frequency = FrequencyLoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.charbonnier(prediction, target)
        if self.ssim_weight:
            loss = loss + self.ssim_weight * self.ssim(prediction, target)
        if self.frequency_weight:
            loss = loss + self.frequency_weight * self.frequency(prediction, target)
        return loss
