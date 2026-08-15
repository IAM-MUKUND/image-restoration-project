"""Compact three-stage progressive restoration network adapted for 2x SR."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.PReLU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.body(x)


class CAB(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            ChannelAttention(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class SupervisedAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.feature = nn.Conv2d(channels, channels, 3, padding=1)
        self.image = nn.Conv2d(channels, 1, 3, padding=1)
        self.attention = nn.Conv2d(1, channels, 3, padding=1)

    def forward(self, features: torch.Tensor, base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.image(features) + base
        attention = torch.sigmoid(self.attention(image))
        return features + self.feature(features) * attention, image


class Stage(nn.Module):
    def __init__(self, channels: int, input_channels: int, blocks: int):
        super().__init__()
        self.head = nn.Conv2d(input_channels, channels, 3, padding=1)
        self.body = nn.Sequential(*(CAB(channels) for _ in range(blocks)))
        self.sam = SupervisedAttention(channels)

    def forward(self, x: torch.Tensor, base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(self.head(x))
        return self.sam(features, base)


class MPRNetSR(nn.Module):
    def __init__(self, channels: int = 32, blocks_per_stage: int = 4):
        super().__init__()
        self.stage1 = Stage(channels, 1, blocks_per_stage)
        self.stage2 = Stage(channels, channels + 2, blocks_per_stage)
        self.stage3_head = nn.Conv2d(channels + 2, channels, 3, padding=1)
        self.stage3_body = nn.Sequential(*(CAB(channels) for _ in range(blocks_per_stage)))
        self.stage3_tail = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        features1, image1 = self.stage1(base, base)
        features2, image2 = self.stage2(torch.cat((base, image1, features1), dim=1), base)
        features3 = self.stage3_body(self.stage3_head(torch.cat((base, image2, features2), dim=1)))
        return self.stage3_tail(features3) + base
