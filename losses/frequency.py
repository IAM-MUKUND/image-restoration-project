from __future__ import annotations

import torch
from torch import nn


class FrequencyLoss(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        height, width = prediction.shape[-2:]
        window_y = torch.hann_window(height, periodic=False, device=prediction.device)
        window_x = torch.hann_window(width, periodic=False, device=prediction.device)
        window = (window_y[:, None] * window_x[None, :]).view(1, 1, height, width)
        pred_fft = torch.fft.rfft2(prediction.float() * window, norm="ortho")
        target_fft = torch.fft.rfft2(target.float() * window, norm="ortho")
        complex_distance = torch.abs(pred_fft - target_fft).mean()
        magnitude_distance = (
            torch.log1p(torch.abs(pred_fft)) - torch.log1p(torch.abs(target_fft))
        ).abs().mean()
        return complex_distance + 0.25 * magnitude_distance
