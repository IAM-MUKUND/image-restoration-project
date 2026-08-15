"""Compact ESRGAN RRDB generator trained in fidelity/PSNR mode for inspection."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    def __init__(self, channels: int, growth: int):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv2d(channels + index * growth, growth, 3, padding=1)
            for index in range(4)
        )
        self.final = nn.Conv2d(channels + 4 * growth, channels, 3, padding=1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for conv in self.convs:
            features.append(self.activation(conv(torch.cat(features, dim=1))))
        return x + 0.2 * self.final(torch.cat(features, dim=1))


class RRDB(nn.Module):
    def __init__(self, channels: int, growth: int):
        super().__init__()
        self.blocks = nn.Sequential(*(ResidualDenseBlock(channels, growth) for _ in range(3)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.2 * self.blocks(x)


class ESRGANGeneratorSR(nn.Module):
    """RRDBNet generator only; no adversarial loss is used in the fair baseline."""

    def __init__(self, channels: int = 32, growth: int = 16, rrdb_blocks: int = 6):
        super().__init__()
        self.first = nn.Conv2d(1, channels, 3, padding=1)
        self.trunk = nn.Sequential(*(RRDB(channels, growth) for _ in range(rrdb_blocks)))
        self.trunk_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.hr = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        shallow = self.first(x)
        features = shallow + self.trunk_conv(self.trunk(shallow))
        return self.hr(self.upsample(features)) + base
