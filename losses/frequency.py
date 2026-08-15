from __future__ import annotations

import torch
from torch import nn


class FrequencyLoss(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.rfft2(prediction, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        return torch.mean(torch.abs(pred_fft - target_fft))
