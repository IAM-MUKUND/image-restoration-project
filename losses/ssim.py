from __future__ import annotations

import torch
from torch import nn
from pytorch_msssim import ssim


class SSIMLoss(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - ssim(prediction, target, data_range=1.0, size_average=True)
