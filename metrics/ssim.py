from __future__ import annotations

import torch
from pytorch_msssim import ssim


def calculate_ssim(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    if prediction.ndim == 3:
        prediction = prediction.unsqueeze(0)
        target = target.unsqueeze(0)
    return float(ssim(prediction, target, data_range=1.0, size_average=True).item())
