"""Inference-time geometric self-ensemble for restoration models."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class GeometricSelfEnsemble(nn.Module):
    """Average inverse-mapped predictions over the dihedral image transforms.

    The wrapper has no trainable parameters.  It exploits small orientation
    biases learned by convolutional restoration models without changing the
    checkpoint.  ``transforms=4`` uses rotations; ``transforms=8`` adds a
    mirrored copy of every rotation.
    """

    def __init__(self, model: nn.Module, transforms: int = 8):
        super().__init__()
        if transforms not in (1, 4, 8):
            raise ValueError("transforms must be one of 1, 4, or 8")
        self.model = model
        self.transforms = transforms

    @staticmethod
    def _forward_transform(image: torch.Tensor, rotation: int, mirror: bool) -> torch.Tensor:
        transformed = torch.rot90(image, rotation, dims=(-2, -1))
        return torch.flip(transformed, dims=(-1,)) if mirror else transformed

    @staticmethod
    def _inverse_transform(image: torch.Tensor, rotation: int, mirror: bool) -> torch.Tensor:
        transformed = torch.flip(image, dims=(-1,)) if mirror else image
        return torch.rot90(transformed, -rotation, dims=(-2, -1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.transforms == 1:
            return self.model(image)

        mirrors = (False, True) if self.transforms == 8 else (False,)
        predictions = []
        for mirror in mirrors:
            for rotation in range(4):
                transformed = self._forward_transform(image, rotation, mirror)
                prediction = self.model(transformed)
                predictions.append(self._inverse_transform(prediction, rotation, mirror))
        return torch.stack(predictions, dim=0).mean(dim=0)


class ResidualCalibrator(nn.Module):
    """Apply a fixed gain/bias to the learned residual over bicubic input."""

    def __init__(self, model: nn.Module, residual_gain: float = 1.0, bias: float = 0.0):
        super().__init__()
        self.model = model
        self.residual_gain = float(residual_gain)
        self.bias = float(bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        prediction = self.model(image)
        base = F.interpolate(image, scale_factor=2, mode="bicubic", align_corners=False)
        return base + self.residual_gain * (prediction - base) + self.bias
